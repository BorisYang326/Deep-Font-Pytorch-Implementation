def label2font(font_book_path: str, label: int) -> str:
    with open(font_book_path, 'r') as f:
        font_families = f.read().splitlines()
    return font_families[label]
