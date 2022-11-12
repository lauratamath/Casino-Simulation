import json

cells = json.load(open('rouletteCells.json'))
bets = json.load(open('bets.json'))

betsCells = [bet["cells"] for bet in bets["bets"] if bet["type"] == "pleno"]
betsNumbers = []
for bet in betsCells:
    betNumbers = []
    for cell in bet:
        number = [_cell["number"] for _cell in cells["cells"] if _cell["position"] == cell][0]
        betNumbers.append(number)
    betsNumbers.append(betNumbers)

# print pretty json
print(json.dumps(betsNumbers, indent=4))