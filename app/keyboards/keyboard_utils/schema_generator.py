from collections.abc import Sequence
from typing import TypeVar, List

T = TypeVar("T")


def create_keyboard_layout(buttons: Sequence[T], count: Sequence[int]) -> List[List[T]]:
    if sum(count) != len(buttons):
        raise ValueError("Количество кнопок не совпадает со схемой")

    layout: List[List[T]] = []
    btn_index = 0

    for row_count in count:
        row = buttons[btn_index:btn_index + row_count]
        layout.append(row)
        btn_index += row_count

    return layout
