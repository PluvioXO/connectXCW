"""Constants for the Connect 4 game engine."""

ROWS: int = 6
COLS: int = 7
WIN_LENGTH: int = 4

# Number of bits per column in the bitboard (rows + 1 sentinel bit)
BITS_PER_COL: int = ROWS + 1  # 7

# Total bits in the bitboard representation
TOTAL_BITS: int = BITS_PER_COL * COLS  # 49

# Mask for a single column (6 actual rows, no sentinel)
COLUMN_MASK: int = (1 << ROWS) - 1  # 0b111111 = 63

# Bottom row mask: bit 0 of each column
BOTTOM_MASK: int = sum(1 << (col * BITS_PER_COL) for col in range(COLS))

# Full board mask: all valid (non-sentinel) bits set
BOARD_MASK: int = BOTTOM_MASK * COLUMN_MASK

# Player identifiers
PLAYER_1: int = 1
PLAYER_2: int = 2
