import json

import pytest

from buc_factory.agent.tool import make_tools


@pytest.fixture
def tools(tmp_path):
    _, dispatch = make_tools(tmp_path)
    return dispatch, tmp_path


# ── write_file ────────────────────────────────────────────────────


def test_write_file_creates_file(tools):
    dispatch, base = tools
    dispatch["write_file"]("hello.txt", "hello world")
    assert (base / "hello.txt").read_text() == "hello world"


def test_write_file_creates_parent_dirs(tools):
    dispatch, base = tools
    dispatch["write_file"]("a/b/c.txt", "nested")
    assert (base / "a" / "b" / "c.txt").read_text() == "nested"


def test_write_file_overwrites_existing(tools):
    dispatch, base = tools
    dispatch["write_file"]("f.txt", "first")
    dispatch["write_file"]("f.txt", "second")
    assert (base / "f.txt").read_text() == "second"


def test_write_file_missing_content_returns_error(tools):
    dispatch, _ = tools
    result = dispatch["write_file"]("out.txt")
    assert result.startswith("ERROR")


# ── read_file ─────────────────────────────────────────────────────


def test_read_file_existing(tools):
    dispatch, base = tools
    (base / "data.txt").write_text("content here")
    assert dispatch["read_file"]("data.txt") == "content here"


def test_read_file_missing_returns_error(tools):
    dispatch, _ = tools
    result = dispatch["read_file"]("nonexistent.txt")
    assert result.startswith("ERROR")


# ── list_files ────────────────────────────────────────────────────


def test_list_files_root(tools):
    dispatch, base = tools
    (base / "a.txt").write_text("a")
    (base / "sub").mkdir()
    (base / "sub" / "b.txt").write_text("b")
    result = dispatch["list_files"](".")
    assert "a.txt" in result
    assert "sub/b.txt" in result


def test_list_files_subdir(tools):
    dispatch, base = tools
    (base / "sub").mkdir()
    (base / "sub" / "x.txt").write_text("x")
    result = dispatch["list_files"]("sub")
    assert "sub/x.txt" in result


def test_list_files_missing_dir_returns_error(tools):
    dispatch, _ = tools
    result = dispatch["list_files"]("nonexistent")
    assert result.startswith("ERROR")


# ── run_python ────────────────────────────────────────────────────


def test_run_python_executes_script(tools):
    dispatch, base = tools
    (base / "hello.py").write_text("print('hello from script')")
    result = dispatch["run_python"]("hello.py")
    assert "hello from script" in result
    assert "exit=0" in result


def test_run_python_captures_stderr(tools):
    dispatch, base = tools
    (base / "err.py").write_text("import sys; sys.stderr.write('oops\\n')")
    result = dispatch["run_python"]("err.py")
    assert "oops" in result


def test_run_python_nonzero_exit(tools):
    dispatch, base = tools
    (base / "fail.py").write_text("raise SystemExit(1)")
    result = dispatch["run_python"]("fail.py")
    assert "exit=1" in result


def test_run_python_missing_script_returns_error(tools):
    dispatch, _ = tools
    result = dispatch["run_python"]("ghost.py")
    assert result.startswith("ERROR")


def test_run_python_path_traversal_blocked(tools, tmp_path):
    dispatch, base = tools
    outside = tmp_path.parent / "evil.py"
    outside.write_text("print('should not run')")
    result = dispatch["run_python"]("../evil.py")
    assert result.startswith("ERROR")


# ── validate_json ─────────────────────────────────────────────────


def test_validate_json_valid(tools):
    dispatch, base = tools
    (base / "data.json").write_text('{"key": "value"}')
    result = dispatch["validate_json"]("data.json")
    assert result.startswith("OK")


def test_validate_json_invalid(tools):
    dispatch, base = tools
    (base / "bad.json").write_text("{not valid json}")
    result = dispatch["validate_json"]("bad.json")
    assert result.startswith("FAIL")


# ── validate_csv_integrity ────────────────────────────────────────


def test_validate_csv_integrity_clean(tools):
    pytest.importorskip("pandas")
    dispatch, base = tools
    (base / "facts.csv").write_text("id,cat_id\n1,10\n2,11\n")
    (base / "dims.csv").write_text("cat_id,name\n10,A\n11,B\n")
    spec = json.dumps(
        {"facts": "facts.csv", "fact_fk": "cat_id", "dim": "dims.csv", "dim_pk": "cat_id"}
    )
    result = dispatch["validate_csv_integrity"](spec)
    assert result.startswith("OK")


def test_validate_csv_integrity_orphans(tools):
    pytest.importorskip("pandas")
    dispatch, base = tools
    (base / "facts.csv").write_text("id,cat_id\n1,10\n2,99\n")  # 99 is orphan
    (base / "dims.csv").write_text("cat_id,name\n10,A\n11,B\n")
    spec = json.dumps(
        {"facts": "facts.csv", "fact_fk": "cat_id", "dim": "dims.csv", "dim_pk": "cat_id"}
    )
    result = dispatch["validate_csv_integrity"](spec)
    assert result.startswith("FAIL")


# ── mark_subtask_complete ─────────────────────────────────────────


def test_mark_subtask_complete(tools):
    dispatch, _ = tools
    result = dispatch["mark_subtask_complete"]("all done")
    assert "all done" in result
