"""Tests for the dashboard password gate.

The Streamlit widgets in `require_password` are not unit tested -- the rest of
`app/` follows the same rule. What *is* tested here is the part that decides
whether a viewer gets in, because the failure mode is an unprotected dashboard
on a public URL and it is silent when it happens.

Most of these are "refuses" rather than "allows". That is deliberate: the bug
worth catching is a gate that falls open, not one that keeps someone out.
"""

from __future__ import annotations

from pathlib import Path

from app.auth import ENV_VAR, configured_password, password_matches

# ------------------------------------------------------------------ reading


def test_reads_the_password_from_the_environment(tmp_path: Path) -> None:
    got = configured_password({ENV_VAR: "hunter2"}, tmp_path / "absent")
    assert got == "hunter2"


def test_reads_the_password_from_the_file_when_env_is_unset(tmp_path: Path) -> None:
    f = tmp_path / "pw"
    f.write_text("from-the-file\n")  # trailing newline on purpose
    assert configured_password({}, f) == "from-the-file"


def test_environment_wins_over_the_file(tmp_path: Path) -> None:
    f = tmp_path / "pw"
    f.write_text("from-the-file")
    assert configured_password({ENV_VAR: "from-the-env"}, f) == "from-the-env"


def test_surrounding_whitespace_is_stripped(tmp_path: Path) -> None:
    # An invisible trailing newline would otherwise make every correct
    # password fail, and the user would have no way to see why.
    f = tmp_path / "pw"
    f.write_text("  spaced  \n")
    assert configured_password({}, f) == "spaced"


# ------------------------------------------------------- refusing to configure


def test_no_env_and_no_file_means_not_configured(tmp_path: Path) -> None:
    assert configured_password({}, tmp_path / "does-not-exist") is None


def test_empty_file_means_not_configured(tmp_path: Path) -> None:
    f = tmp_path / "pw"
    f.write_text("")
    assert configured_password({}, f) is None


def test_whitespace_only_file_means_not_configured(tmp_path: Path) -> None:
    f = tmp_path / "pw"
    f.write_text("   \n\t\n")
    assert configured_password({}, f) is None


def test_empty_env_var_falls_through_to_the_file(tmp_path: Path) -> None:
    f = tmp_path / "pw"
    f.write_text("file-password")
    assert configured_password({ENV_VAR: ""}, f) == "file-password"


def test_a_directory_where_the_file_should_be_means_not_configured(
    tmp_path: Path,
) -> None:
    d = tmp_path / "pw"
    d.mkdir()
    assert configured_password({}, d) is None


# -------------------------------------------------------------- matching


def test_the_right_password_matches() -> None:
    assert password_matches("correct-horse", "correct-horse") is True


def test_a_wrong_password_does_not_match() -> None:
    assert password_matches("wrong", "correct-horse") is False


def test_matching_is_case_sensitive() -> None:
    assert password_matches("Correct-Horse", "correct-horse") is False


def test_a_prefix_of_the_password_does_not_match() -> None:
    assert password_matches("correct", "correct-horse") is False


def test_an_empty_expected_password_matches_nothing() -> None:
    # The dangerous case: a blank configured password must not turn the gate
    # into a door anyone can walk through by submitting an empty form.
    assert password_matches("", "") is False
    assert password_matches("anything", "") is False


def test_an_empty_entry_does_not_match_a_real_password() -> None:
    assert password_matches("", "correct-horse") is False


def test_non_ascii_passwords_compare_correctly() -> None:
    # `hmac.compare_digest` needs bytes; a naive encode of a non-ASCII string
    # is the kind of thing that raises at exactly the wrong moment.
    assert password_matches("paswórd-é", "paswórd-é") is True
    assert password_matches("paswórd-e", "paswórd-é") is False
