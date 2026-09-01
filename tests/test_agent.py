# -*- coding: utf-8 -*-
"""Unit tests for the agent's own logic (sandbox, tools, context, parsing).

Run with:  python tests/test_agent.py
Uses only the standard library (unittest).
"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from coding_agent.agent import Agent
from coding_agent.config import Config
from coding_agent.context import Context
from coding_agent.sessions import list_sessions, load_session, save_session
from coding_agent.tools import ToolBox


def make_toolbox(tmpdir: str) -> ToolBox:
    cfg = Config(workspace=Path(tmpdir))
    return ToolBox(cfg)


class TestSandbox(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.box = make_toolbox(self.tmp.name)
        self.ws = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_escape_via_parent(self):
        out = self.box.read_file("../outside.txt")
        self.assertIn("outside the workspace", out)

    def test_escape_via_absolute(self):
        out = self.box.write_file(str(Path.home() / "evil.txt"), "x")
        self.assertIn("outside the workspace", out)

    def test_write_read_roundtrip(self):
        r1 = self.box.write_file("a/b/c.txt", "hello 世界")
        self.assertIn("wrote", r1)
        self.assertEqual((self.ws / "a" / "b" / "c.txt").read_text(encoding="utf-8"),
                         "hello 世界")
        r2 = self.box.read_file("a/b/c.txt")
        self.assertIn("hello 世界", r2)

    def test_edit_file(self):
        self.box.write_file("d.txt", "one two three")
        self.assertEqual(self.box.edit_file("d.txt", "two", "TWO"),
                         "edit applied successfully")
        self.assertEqual((self.ws / "d.txt").read_text(encoding="utf-8"),
                         "one TWO three")
        self.assertIn("not found", self.box.edit_file("d.txt", "zzz", "y"))

    def test_unknown_tool(self):
        out = self.box.execute("nope", {})
        self.assertIn("unknown tool", out)


class TestSkills(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "python-testing" / "SKILL.md").parent.mkdir(parents=True)
        (root / "python-testing" / "SKILL.md").write_text(
            "---\nname: python-testing\n"
            "description: write & run unittest tests\n"
            "---\nBody of skill.\n", encoding="utf-8")
        (root / "no-frontmatter" / "SKILL.md").parent.mkdir()
        (root / "no-frontmatter" / "SKILL.md").write_text(
            "Just a body without frontmatter.\n", encoding="utf-8")
        (root / "not-a-skill").mkdir()
        (root / "not-a-skill" / "README.txt").write_text("x", encoding="utf-8")
        self.cfg = Config(workspace=Path(self.tmp.name) / "ws",
                          skills_dir=root)
        self.box = ToolBox(self.cfg)

    def tearDown(self):
        self.tmp.cleanup()

    def test_list_skills(self):
        out = self.box.list_skills()
        self.assertIn("python-testing", out)
        self.assertIn("write & run unittest tests", out)
        # dir without SKILL.md must not be listed
        self.assertNotIn("not-a-skill", out)

    def test_load_skill_returns_body(self):
        out = self.box.load_skill("python-testing")
        self.assertIn("Body of skill.", out)
        # frontmatter stripped from returned content
        self.assertNotIn("write & run unittest", out)

    def test_load_unknown_skill(self):
        out = self.box.load_skill("nope")
        self.assertIn("unknown skill", out)

    def test_parse_fallback_without_frontmatter(self):
        out = self.box.load_skill("no-frontmatter")
        self.assertIn("Just a body", out)


class TestRunCommand(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.box = make_toolbox(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_runs_in_workspace(self):
        self.box.write_file("hello.txt", "hi")
        out = self.box.run_command("type hello.txt" if sys.platform == "win32"
                                   else "cat hello.txt")
        self.assertIn("hi", out)
        self.assertIn("exit code 0", out)


class TestContext(unittest.TestCase):
    def test_compaction_keeps_recent(self):
        # Small enough messages that keep_recent of them fit inside the budget.
        ctx = Context(system="sys", char_budget=400, keep_recent=4)
        for i in range(20):
            ctx.append({"role": "user", "content": f"message number {i} " * 4})
            ctx.append({"role": "assistant", "content": f"answer {i} " * 4})
        # After compaction, history must start with a compressed note and end
        # with the most recent messages, and stay near the budget afterwards.
        self.assertEqual(ctx.history[0]["role"], "system")
        self.assertLessEqual(len(ctx.history), 1 + ctx.keep_recent)
        self.assertLessEqual(ctx.summary_stats()["approx_chars"],
                             ctx.char_budget + 150)

    def test_tool_roundtrip_not_cut(self):
        ctx = Context(system="sys", char_budget=100, keep_recent=2)
        ctx.append({"role": "assistant", "content": "call it",
                    "tool_calls": [{"id": "x", "function": {"name": "f", "arguments": "{}"}}]})
        ctx.append({"role": "tool", "tool_call_id": "x", "content": "result " * 50})
        # The tool result must survive (it is the tail), never orphaned.
        self.assertEqual(ctx.history[-1]["role"], "tool")


class TestArgParsing(unittest.TestCase):
    def test_strict_json(self):
        args, err = Agent._parse_args("f", '{"path": "x.py"}')
        self.assertIsNone(err)
        self.assertEqual(args["path"], "x.py")

    def test_loose_json_with_noise(self):
        args, err = Agent._parse_args("f", 'Here you go {"path": "x.py"} ok?')
        self.assertIsNone(err)
        self.assertEqual(args["path"], "x.py")

    def test_garbage(self):
        args, err = Agent._parse_args("f", "not json at all")
        self.assertIsNotNone(err)


class _FinalLLM:
    """Scripted LLM that just answers immediately (no tools)."""

    def chat(self, messages, tools):
        return {"role": "assistant", "content": "ok"}


class TestSessions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.logs = self.root / "logs"

    def tearDown(self):
        self.tmp.cleanup()

    def test_save_load_roundtrip(self):
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "read_file",
                                          "arguments": '{"path": "a.txt"}'}}]},
            {"role": "tool", "tool_call_id": "c1", "name": "read_file",
             "content": "file body 世界"},
        ]
        path = save_session(history, "some task", "done", self.logs)
        self.assertTrue(path.is_file())
        data = load_session(path)
        self.assertEqual(data["history"], history)
        self.assertEqual(data["final_answer"], "done")

    def test_list_sessions_newest_first(self):
        save_session([{"role": "user", "content": "a"}], "t1", "a1", self.logs)
        save_session([{"role": "user", "content": "b"}], "t2", "b2", self.logs)
        listed = list_sessions(self.logs)
        self.assertEqual(len(listed), 2)
        # Files are timestamped, so a descending sort == newest first.
        self.assertEqual(load_session(listed[0])["task"], "t2")

    def test_resume_keeps_prior_history(self):
        cfg = Config(workspace=self.root / "ws")
        agent = Agent(cfg, llm=_FinalLLM())
        agent.run("first task", stream=False)
        contents = [m.get("content") for m in agent.context.history]
        self.assertIn("first task", contents)

        # Reload into a fresh agent exactly as `--resume` would.
        path = save_session(agent.context.history, "first task", "ok", self.logs)
        fresh = Agent(cfg, llm=_FinalLLM())
        fresh.load_history(load_session(path)["history"])
        answer = fresh.continue_run("follow up", stream=False)
        self.assertEqual(answer, "ok")
        contents = [m.get("content") for m in fresh.context.history]
        self.assertIn("first task", contents)  # old conversation kept
        self.assertIn("follow up", contents)   # new user message appended


if __name__ == "__main__":
    unittest.main(verbosity=2)
