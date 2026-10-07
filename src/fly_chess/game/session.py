from uuid import uuid4

import chess

from fly_chess.core.rules import ChessGame


class HumanSession:
    def __init__(self, human_color=chess.WHITE, game=None):
        self.game = game or ChessGame(claim_draws=False)
        self.human_color = human_color
        self.game_id = uuid4().hex
        self.revision = 0
        self.manual_result = None

    @property
    def token(self):
        return self.game_id, self.revision

    @property
    def finished(self):
        return self.manual_result is not None or self.game.result is not None

    @property
    def human_turn(self):
        return not self.finished and self.game.board.turn == self.human_color

    def invalidate(self):
        self.revision += 1

    def move_human(self, move):
        if not self.human_turn:
            raise ValueError("It is not the human turn")
        self.game.push(move)
        self.invalidate()

    def apply_ai(self, token, move):
        if token != self.token or self.finished or self.human_turn:
            return False
        self.game.push(move)
        self.invalidate()
        return True

    def undo(self):
        if not self.can_undo:
            return False
        self.manual_result = None
        while self.game.board.move_stack:
            self.game.undo()
            if self.game.board.turn == self.human_color:
                break
        self.invalidate()
        return True

    @property
    def can_undo(self):
        root = self.game.board.root()
        return any((root.turn if i % 2 == 0 else not root.turn) == self.human_color
                   for i in range(len(self.game.board.move_stack)))

    def resign(self):
        if not self.finished:
            self.manual_result = "You resigned. Fly wins."
            self.invalidate()

    def claim_draw(self):
        if self.human_turn and self.game.board.can_claim_draw():
            self.manual_result = "Draw claimed."
            self.invalidate()
            return True
        return False

    def history(self):
        board = self.game.board.root()
        moves = []
        for move in self.game.board.move_stack:
            moves.append(board.san(move))
            board.push(move)
        return moves

    def captured(self):
        board = self.game.board.root()
        pieces = {chess.WHITE: [], chess.BLACK: []}
        for move in self.game.board.move_stack:
            if board.is_en_passant(move):
                pieces[board.turn].append(chess.PAWN)
            elif board.piece_at(move.to_square):
                pieces[board.turn].append(board.piece_type_at(move.to_square))
            board.push(move)
        return pieces
