import chess
import pygame as pg
from fly_chess.ui.theme import INK, MUTED, piece_image

BOARD = pg.Rect(56, 194, 608, 608)


def square_rect(square, flipped=False):
    file, rank = chess.square_file(square), chess.square_rank(square)
    col, row = (7-file, rank) if flipped else (file, 7-rank)
    return pg.Rect(BOARD.x+col*76, BOARD.y+row*76, 76, 76)


def square_at(pos, flipped=False):
    if not BOARD.collidepoint(pos):
        return None
    col, row = (int(pos[0])-BOARD.x)//76, (int(pos[1])-BOARD.y)//76
    return chess.square(7-col, row) if flipped else chess.square(col, 7-row)


def draw_board(p, board, flipped=False, selected=None, cursor=None, animation=None):
    last = board.peek() if board.move_stack else None
    legal = {m.to_square for m in board.legal_moves if m.from_square == selected}
    for square in chess.SQUARES:
        rect = square_rect(square, flipped)
        light = (chess.square_file(square)+chess.square_rank(square)) % 2 == 1
        p.rect((217, 216, 209) if light else (102, 103, 99), rect)
        if last and square in (last.from_square, last.to_square):
            p.rect((245, 244, 237), rect.inflate(-7, -7), 2)
        if square == selected:
            p.rect((20, 20, 20), rect.inflate(-3, -3), 4)
        if board.is_check() and square == board.king(board.turn):
            for inset in (9, 14):
                p.rect((20, 20, 20), rect.inflate(-inset*2, -inset*2), 2)
        piece = board.piece_at(square)
        if piece and not (animation and square == animation[0].to_square):
            p.piece(piece, 72, rect.move(2, 2).topleft)
        if square in legal:
            if piece:
                p.circle((20, 20, 20), rect.center, 34, 3)
            else:
                p.circle((20, 20, 20), rect.center, 6)
        if square == cursor:
            p.rect((245, 244, 237), rect.inflate(-2, -2), 1)
    if animation:
        move, piece, progress = animation
        start, end = square_rect(move.from_square, flipped), square_rect(move.to_square, flipped)
        xy = (start.x+(end.x-start.x)*progress+2, start.y+(end.y-start.y)*progress+2)
        p.piece(piece, 72, xy)
    for i in range(8):
        p.text('hgfedcba'[i] if flipped else 'abcdefgh'[i], 89+i*76, 807, 14, MUTED, True)
        p.text(str(i+1 if flipped else 8-i), 33, 221+i*76, 14, MUTED, True)
