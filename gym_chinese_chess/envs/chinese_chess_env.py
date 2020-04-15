from __future__ import print_function
import gym
import re, sys, time
from gym import error, spaces, utils
from gym.utils import seeding
import copy
from itertools import count
from collections import namedtuple
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import os

piece = {'P': 44, 'N': 108, 'B': 23, 'R': 233, 'A': 23, 'C': 101, 'K': 2500}

A0, I0, A9, I9 = 12 * 16 + 3, 12 * 16 + 11, 3 * 16 + 3, 3 * 16 + 11

initial = (
    '               \n'  # 0 -  9
    '               \n'  # 10 - 19
    '               \n'  # 10 - 19
    '   rnbakabnr   \n'  # 20 - 29
    '   .........   \n'  # 40 - 49
    '   .c.....c.   \n'  # 40 - 49
    '   p.p.p.p.p   \n'  # 30 - 39
    '   .........   \n'  # 50 - 59
    '   .........   \n'  # 70 - 79
    '   P.P.P.P.P   \n'  # 80 - 89
    '   .C.....C.   \n'  # 70 - 79
    '   .........   \n'  # 70 - 79
    '   RNBAKABNR   \n'  # 90 - 99
    '               \n'  # 100 -109
    '               \n'  # 100 -109
    '               \n'  # 110 -119
)

pst = {
    "P": (
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 9, 9, 9, 11, 13, 11, 9, 9, 9, 0, 0, 0, 0,
        0, 0, 0, 19, 24, 34, 42, 44, 42, 34, 24, 19, 0, 0, 0, 0,
        0, 0, 0, 19, 24, 32, 37, 37, 37, 32, 24, 19, 0, 0, 0, 0,
        0, 0, 0, 19, 23, 27, 29, 30, 29, 27, 23, 19, 0, 0, 0, 0,
        0, 0, 0, 14, 18, 20, 27, 29, 27, 20, 18, 14, 0, 0, 0, 0,
        0, 0, 0, 7, 0, 13, 0, 16, 0, 13, 0, 7, 0, 0, 0, 0,
        0, 0, 0, 7, 0, 7, 0, 15, 0, 7, 0, 7, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 11, 15, 11, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ),
    "B": (
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 40, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 38, 0, 0, 40, 43, 40, 0, 0, 38, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 43, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 40, 40, 0, 40, 40, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ),
    "N": (
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 90, 90, 90, 96, 90, 96, 90, 90, 90, 0, 0, 0, 0,
        0, 0, 0, 90, 96, 103, 97, 94, 97, 103, 96, 90, 0, 0, 0, 0,
        0, 0, 0, 92, 98, 99, 103, 99, 103, 99, 98, 92, 0, 0, 0, 0,
        0, 0, 0, 93, 108, 100, 107, 100, 107, 100, 108, 93, 0, 0, 0, 0,
        0, 0, 0, 90, 100, 99, 103, 104, 103, 99, 100, 90, 0, 0, 0, 0,
        0, 0, 0, 90, 98, 101, 102, 103, 102, 101, 98, 90, 0, 0, 0, 0,
        0, 0, 0, 92, 94, 98, 95, 98, 95, 98, 94, 92, 0, 0, 0, 0,
        0, 0, 0, 93, 92, 94, 95, 92, 95, 94, 92, 93, 0, 0, 0, 0,
        0, 0, 0, 85, 90, 92, 93, 78, 93, 92, 90, 85, 0, 0, 0, 0,
        0, 0, 0, 88, 85, 90, 88, 90, 88, 90, 85, 88, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ),
    "R": (
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 206, 208, 207, 213, 214, 213, 207, 208, 206, 0, 0, 0, 0,
        0, 0, 0, 206, 212, 209, 216, 233, 216, 209, 212, 206, 0, 0, 0, 0,
        0, 0, 0, 206, 208, 207, 214, 216, 214, 207, 208, 206, 0, 0, 0, 0,
        0, 0, 0, 206, 213, 213, 216, 216, 216, 213, 213, 206, 0, 0, 0, 0,
        0, 0, 0, 208, 211, 211, 214, 215, 214, 211, 211, 208, 0, 0, 0, 0,
        0, 0, 0, 208, 212, 212, 214, 215, 214, 212, 212, 208, 0, 0, 0, 0,
        0, 0, 0, 204, 209, 204, 212, 214, 212, 204, 209, 204, 0, 0, 0, 0,
        0, 0, 0, 198, 208, 204, 212, 212, 212, 204, 208, 198, 0, 0, 0, 0,
        0, 0, 0, 200, 208, 206, 212, 200, 212, 206, 208, 200, 0, 0, 0, 0,
        0, 0, 0, 194, 206, 204, 212, 200, 212, 204, 206, 194, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ),
    "C": (
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 100, 100, 96, 91, 90, 91, 96, 100, 100, 0, 0, 0, 0,
        0, 0, 0, 98, 98, 96, 92, 89, 92, 96, 98, 98, 0, 0, 0, 0,
        0, 0, 0, 97, 97, 96, 91, 92, 91, 96, 97, 97, 0, 0, 0, 0,
        0, 0, 0, 96, 99, 99, 98, 100, 98, 99, 99, 96, 0, 0, 0, 0,
        0, 0, 0, 96, 96, 96, 96, 100, 96, 96, 96, 96, 0, 0, 0, 0,
        0, 0, 0, 95, 96, 99, 96, 100, 96, 99, 96, 95, 0, 0, 0, 0,
        0, 0, 0, 96, 96, 96, 96, 96, 96, 96, 96, 96, 0, 0, 0, 0,
        0, 0, 0, 97, 96, 100, 99, 101, 99, 100, 96, 97, 0, 0, 0, 0,
        0, 0, 0, 96, 97, 98, 98, 98, 98, 98, 97, 96, 0, 0, 0, 0,
        0, 0, 0, 96, 96, 97, 99, 99, 99, 97, 96, 96, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    )
}

pst["A"] = pst["B"]
pst["K"] = pst["P"]
pst["K"] = [i + piece["K"] if i > 0 else 0 for i in pst["K"]]

# Lists of possible moves for each piece type.
N, E, S, W = -16, 1, 16, -1
directions = {
    'P': (N, W, E),
    'N': (N + N + E, E + N + E, E + S + E, S + S + E, S + S + W, W + S + W, W + N + W, N + N + W),
    'B': (2 * N + 2 * E, 2 * S + 2 * E, 2 * S + 2 * W, 2 * N + 2 * W),
    'R': (N, E, S, W),
    'C': (N, E, S, W),
    'A': (N + E, S + E, S + W, N + W),
    'K': (N, E, S, W)
}

MATE_LOWER = piece['K'] - (
        2 * piece['R'] + 2 * piece['N'] + 2 * piece['B'] + 2 * piece['A'] + 2 * piece['C'] + 5 * piece['P'])
MATE_UPPER = piece['K'] + (
        2 * piece['R'] + 2 * piece['N'] + 2 * piece['B'] + 2 * piece['A'] + 2 * piece['C'] + 5 * piece['P'])


class Position(namedtuple('Position', 'board')):
    """ A state of a chess game
    board -- a 256 char representation of the board
    """

    def gen_moves(self):
        # For each of our pieces, iterate through each possible 'ray' of moves,
        # as defined in the 'directions' map. The rays are broken e.g. by
        # captures or immediately in case of pieces such as knights.
        for i, p in enumerate(self.board):
            if p == 'K':
                for scanpos in range(i - 16, A9, -16):
                    if self.board[scanpos] == 'k':
                        yield (i, scanpos)
                    elif self.board[scanpos] != '.':
                        break
            if not p.isupper(): continue
            if p == 'C':
                for d in directions[p]:
                    cfoot = 0
                    for j in count(i + d, d):
                        q = self.board[j]
                        if q.isspace(): break
                        if cfoot == 0 and q == '.':
                            yield (i, j)
                        elif cfoot == 0 and q != '.':
                            cfoot += 1
                        elif cfoot == 1 and q.islower():
                            yield (i, j);
                            break
                        elif cfoot == 1 and q.isupper():
                            break;
                continue
            for d in directions[p]:
                for j in count(i + d, d):
                    q = self.board[j]
                    # Stay inside the board, and off friendly pieces
                    if q.isspace() or q.isupper(): break
                    # 过河的卒/兵才能横着走
                    if p == 'P' and d in (E, W) and i > 128:
                        break
                    # j & 15 等价于 j % 16但是更快
                    elif p in ('A', 'K') and (j < 160 or j & 15 > 8 or j & 15 < 6):
                        break
                    elif p == 'B' and j < 128:
                        break
                    elif p == 'N':
                        n_diff_x = (j - i) & 15
                        if n_diff_x == 14 or n_diff_x == 2:
                            if self.board[i + (1 if n_diff_x == 2 else -1)] != '.': break
                        else:
                            if j > i and self.board[i + 16] != '.':
                                break
                            elif j < i and self.board[i - 16] != '.':
                                break
                    elif p == 'B' and self.board[i + d // 2] != '.':
                        break
                    # Move it
                    yield (i, j)
                    # Stop crawlers from sliding, and sliding after captures
                    if p in 'PNBAK' or q.islower(): break

    def rotate(self):
        ''' Rotates the board, preserving enpassant '''
        return Position(
            self.board[-2::-1].swapcase() + " ")

    @staticmethod
    def rotate_board_str(board_str):
        return board_str[-2::-1].swapcase() + " "

    def nullmove(self):
        ''' Like rotate, but clears ep and kp '''
        return self.rotate()

    def move(self, move):
        i, j = move
        p, q = self.board[i], self.board[j]
        put = lambda board, i, p: board[:i] + p + board[i + 1:]
        # Copy variables and reset ep and kp
        board = self.board
        # Actual move
        board = put(board, j, board[i])
        board = put(board, i, '.')
        return Position(board).rotate()

    def value(self, move):
        i, j = move
        p, q = self.board[i], self.board[j]
        # Actual move
        score = pst[p][j] - pst[p][i]
        # Capture
        if q.islower():
            score += pst[q.upper()][255 - j - 1]
        return score

    def player_has_king(self):
        return "K" in self.board

    def oppo_has_king(self):
        return "k" in self.board

    def print_pos(self):
        out_str = "\n"
        uni_pieces = {'R': '车', 'N': '马', 'B': '相', 'A': '仕', 'K': '帅', 'P': '兵', 'C': '炮',
                      'r': '俥', 'n': '傌', 'b': '象', 'a': '士', 'k': '将', 'p': '卒', 'c': '砲', '.': '．'}
        for i, row in enumerate(self.board.split()):
            out_str += "{}{}{}\n".format(' ', 9 - i, ''.join(uni_pieces.get(p, p) for p in row))
        out_str += '  ａｂｃｄｅｆｇｈｉ\n\n'
        return out_str

    def to_numpy(self):
        out_str = "\n"
        uni_pieces = {'R': 1, 'N': 2, 'B': 3, 'A': 4, 'K': 5, 'P': 6, 'C': 7,
                      'r': -1, 'n': -2, 'b': -3, 'a': -4, 'k': -5, 'p': -6, 'c': -7, '.': 0}
        ind = 0
        out_array = np.zeros((10, 9))
        for i, row in enumerate(self.board.split()):
            row = row.replace(" ", "")
            if row:
                out_array[ind] = np.asarray([uni_pieces[j] for j in row])
                ind += 1
        return out_array


class ChineseChessEnv(gym.Env):
    fontText = ImageFont.truetype(os.path.join(os.path.dirname(__file__), "..", "..", "fonts", "msjh.ttf"), 80,
                                  encoding="utf-8")

    def __init__(self):
        self.cache_steps = 6
        self.pos = Position(initial)
        self.his = [copy.copy(self.pos.board)]
        self.posdic = {}
        self.posdic[self.pos.board] = 1

        self.observation_space = spaces.Box(-7, 7, (self.cache_steps, 10, 9))  # board 8x8
        # 棋盘的笛卡尔积 + 投降
        self.action_space = spaces.Discrete(90 * 90)
        self.current_player = 0
        self.resigned = [False, False]
        self.boardcount = {}

    def get_history_positions(self):
        return [Position(i) for i in self.his]

    def generate_observation(self):
        observation = np.zeros([self.cache_steps, 10, 9])
        for i, one_pos in enumerate(self.his[::-1][:self.cache_steps]):
            if i % 2 == 0:
                observation[i] = Position(one_pos).to_numpy()
            else:
                observation[i] = Position(one_pos).rotate().to_numpy()
        return observation

    def cv2ImgAddText(self, draw, text, left, top, textColor=(0, 255, 0)):
        draw.text((left, top), text, textColor, font=ChineseChessEnv.fontText)

    def generate_image(self):
        image = np.ones([1000, 820, 3], dtype=np.uint8)
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)

        font = cv2.FONT_HERSHEY_SIMPLEX

        y0, dy = -10, 80
        text = self.render()
        # Using cv2.putText() method
        for i, one_txt in enumerate(text.split('\n')):
            y = y0 + i * dy
            x0 = -20
            dx = 70
            for j, one_char in enumerate(one_txt):
                x = x0 + j * dx
                if one_char in "．":
                    textColor = (0, 255, 0)
                elif one_char in "1234567890ａｂｃｄｅｆｇｈｉ":
                    textColor = (255, 255, 255)
                elif one_char in "车马相仕帅兵炮":
                    textColor = (255, 0, 0)
                else:
                    textColor = (0, 0, 255)

                self.cv2ImgAddText(draw, one_char, x, y, textColor)

        return np.asarray(image)

    def step(self, action):
        possible_actions = self.get_possible_actions()
        assert (action in possible_actions)
        assert (self.pos.player_has_king())
        action_str = ChineseChessEnv.action2move(action)
        if action_str == "resign":
            assert (self.resigned[self.current_player] != True)
            self.resigned[self.current_player] == True
            reward = -1
            done = True
            info = {"history": self.get_history_positions()}
            return self.generate_observation(), reward, done, info
        else:
            # action str 应该类似b2e2
            if len(action_str) > 4:
                raise RuntimeError(f"action {action} not recognized")
            from_str, to_str = action_str[:2], action_str[2:]
            from_cord, to_cord = ChineseChessEnv.str2cord(from_str), ChineseChessEnv.str2cord(to_str)
            value_diff = self.pos.value((from_cord, to_cord))
            self.pos.value((from_cord, to_cord))
            move_piece = self.pos.board[from_cord]
            self.pos = self.pos.move((from_cord, to_cord))
            self.his.append(copy.copy(self.pos.board))
            self.his = self.his[-6:]
            self.boardcount.setdefault(self.pos.board, 0)
            self.boardcount[self.pos.board] += 1

            reward = 0
            if self.boardcount[self.pos.board] >= 3:
                if move_piece != "K":
                    done = True
                    reward = -1
            elif not self.pos.player_has_king():
                # 这里条件是player has king，但是由于在pos.move中局面被rotate过（红黑交换），所以这里其实在判断这一步完成后是否已经吃掉对方将军
                done = True
                reward = 1
            else:
                done = False
            # 交换红黑方
            self.current_player = 1 - self.current_player
            info = {
                "history": self.get_history_positions(),
                "value": value_diff,
            }
            return self.generate_observation(), reward, done, info

    def reset(self):
        self.pos = Position(initial)
        self.his = [copy.copy(self.pos.board)]
        self.posdic = {self.pos.board: 1}
        self.current_player = 0
        self.resigned = [False, False]
        self.boardcount = {}
        return self.generate_observation()

    def render(self, mode='human'):
        return self.pos.print_pos()

    @staticmethod
    def str2cord(c):
        fil, rank = ord(c[0]) - ord('a'), int(c[1])
        return A0 + fil - 16 * rank

    @staticmethod
    def str2action(c):
        fil, rank = ord(c[0]) - ord('a'), int(c[1])
        return fil + 9 * rank

    @staticmethod
    def cord2str(i):
        rank, fil = divmod(i - A0, 16)
        return chr(fil + ord('a')) + str(-rank)

    @staticmethod
    def action2str(i):
        rank, fil = divmod(i, 9)
        return chr(fil + ord('a')) + str(rank)

    @staticmethod
    def resign_action():
        return 90 * 90

    @staticmethod
    def has_resigned(action):
        return action == ChineseChessEnv.resign_action()

    @staticmethod
    def action2move(action):
        """
        Encode move into action
        """
        if action == ChineseChessEnv.resign_action():
            return 'resign'
        elif action < 90 * 90:
            from_act, to_act = divmod(action, 90)
            return ChineseChessEnv.action2str(from_act) + ChineseChessEnv.action2str(to_act)

    @staticmethod
    def move_to_action(move):
        """
        Encode move into action
        """
        if move == 'resign':
            return ChineseChessEnv.resign_action()
        else:
            match = re.match('([a-i][0-9])' * 2, move)
            if not match:
                raise RuntimeError(f"{match} not recognized")
            from_act, to_act = ChineseChessEnv.str2action(match.group(1)), ChineseChessEnv.str2action(match.group(2))
            move_int = from_act * 90 + to_act
            return move_int

    def get_possible_actions(self):
        moves = self.get_possible_moves()
        return [ChineseChessEnv.move_to_action(m) for m in moves]

    def get_possible_moves(self):
        """
        Returns a list of numpy tuples
        -----
        piece_id - id
        position - (row, ccolumn)
        new_position - (row, column)
        """
        if self.current_player == 0 or self.current_player == 1:
            # 红方情况
            moves = []
            for from_cord, to_cord in self.pos.gen_moves():
                one_move = ChineseChessEnv.cord2str(from_cord) + ChineseChessEnv.cord2str(to_cord)
                moves.append(one_move)
        else:
            raise RuntimeError(f"player {self.current_player} not recognized")
        if not self.pos.player_has_king():
            # 如果将军已经被吃掉，那么输了，同样返回空的move数组
            moves = []
        elif self.resigned[self.current_player]:
            # 如果已经投降或者议和，返回空的move数组
            moves = []
        else:
            pass
            # moves.append("resign")
        return moves
