from smart_dice_rl.boards import CONFIGS, generate_board, validate_board

def test_generated_boards_are_reproducible_and_legal():
    for size,ns,nl,seed in CONFIGS:
        a=generate_board(size,ns,nl,seed); b=generate_board(size,ns,nl,seed)
        assert a==b and len(a['snakes'])==ns and len(a['ladders'])==nl
        validate_board(size,a['snakes'],a['ladders'])
