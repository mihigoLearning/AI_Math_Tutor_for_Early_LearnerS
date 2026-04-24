"""Tests for src/visual.py"""

import pytest

from visual import render, count_blobs, parse_visual, CANVAS, BG


def test_parse_visual_valid():
    assert parse_visual("apples_3") == ("apples", 3)
    assert parse_visual("goats_5") == ("goats", 5)
    assert parse_visual("drums_12") == ("drums", 12)


def test_parse_visual_invalid():
    assert parse_visual("abstract_27_plus_14") is None  # not <name>_<int>
    assert parse_visual("") is None
    assert parse_visual("apples") is None


def test_render_canvas_size():
    img = render("apples_3")
    assert img.size == CANVAS


def test_render_abstract_item_shows_hint_overlay():
    img = render("abstract_1_plus_2")  # no count
    # Abstract items now show a text/card hint instead of plain white.
    assert any(px != BG for px in img.getdata())


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
def test_count_blobs_roundtrip(n):
    img = render(f"item_{n}")
    assert count_blobs(img) == n, f"want {n}, got {count_blobs(img)}"


def test_render_is_deterministic():
    img1 = render("apples_4")
    img2 = render("apples_4")
    # Same pixels
    assert list(img1.getdata()) == list(img2.getdata())


def test_count_blobs_filters_tiny_specks():
    img = render("apples_5")
    # default min_size should still find the 5 real blobs
    assert count_blobs(img, min_size=80) == 5
