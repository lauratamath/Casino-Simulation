import numpy as np
import pandas as pd
import plotly.express as px
from matplotlib import pyplot as plt
from colorama import Fore, Style
import json
import random

CELLS = json.load(open('rouletteCells.json'))["cells"]
BETS = json.load(open('bets.json'))["bets"]

class Roulette():
    def __init__(self, min_bet = 1, max_bet = 100, max_players = 6):
        # cells from rouletteCells.json file
        self.min_bet = min_bet # minimo de apuesta
        self.max_bet = max_bet # maximo de apuesta
        self.bets = [] # Registro de todas las apuestas en la ruleta
        self.max_players = max_players
        self.current_players = []

    # Como un jugador decide su apuesta, monto y tipo
    # TODO: investigar / usar la probabilidad de cada tipo de apuesta
    def set_bet(self, player_no, arriving_time):
        bet = {
            "player_no": player_no,
            "arriving_time": arriving_time,
            "amount": np.random.randint(self.min_bet, self.max_bet),
            "bet": np.random.choice(BETS),
            "result": None,
            "win": None,
            "earnings": None
        }
        self.bets.append(bet)

    def check_space(self):
        return len(self.current_players) < self.max_players

    # Agrega un jugador a la ruleta y su apuesta
    def add_player_and_bet(self, player_no, arriving_time):
        if len(self.current_players) < self.max_players:
            self.current_players.append(player_no)
            self.set_bet(player_no, arriving_time)
            return True
        else:
            return False

    def check_win(self, bet, cell):
        if cell["position"] in bet["cells"]:
            return True
        else:
            return False

    # Se gira la ruleta, se obtienen los resultados de las apuestas y se limpian los jugadores
    def spin(self):
        # Se obtiene un random de la lista CELLS
        result = np.random.choice(CELLS)
        # Se resuelven las apuestas
        for _bet in self.bets:
            if _bet["result"] is None:
                _bet["result"] = result
                _bet["win"] = self.check_win(_bet["bet"], result)
                _bet["earnings"] = _bet["bet"]["payout"] * _bet["amount"] if _bet["win"] else -_bet["amount"]

    def do_player_stay(self, player_no):
        timesPlayed = len([bet for bet in self.bets if bet["player_no"] == player_no])
        timesWon = len([bet for bet in self.bets if bet["player_no"] == player_no and bet["win"]])
        return random.random() <= (timesWon / timesPlayed)

    def reopen_roulette(self):
        players_keep = []
        players_leave = []
        for player_no in self.current_players:
            if self.do_player_stay(player_no):
                players_keep.append(player_no)
            else:
                players_leave.append(player_no)
        self.current_players = players_keep
        spaces_available = self.max_players - len(self.current_players)
        return players_keep, players_leave, spaces_available

class Casino():

    def __init__(self, average_incidence = 0.25, roulette_no = 5, time = 8):
        self.average_incidence = average_incidence  # cantidad máxima de Gamblers por mesa 
        self.roulette_no = roulette_no # cantidad de ruletas que tiene el casino
        self.time = time #cantidad de horas de simulacion
    # Calculo de siguiente llegada (inverse CDF)
    def next_ts(self, t): # No hace gran cosa, pero existe porque los eventos serán procesos de poisson homogeneos
        return t - (np.log(np.random.uniform())/self.average_incidence)
    # Calculo para tiempo
    # ----------------------MODIFICAR PARA PODER CAMBIAR LA CANTIDAD DE TIEMPO DE JUEGO---------------------------
    def get_exponential(self, lamda):
        return -(1 / lamda)*np.log(np.random.uniform())

    def simulate(self):
        t = 0 
        T = 60*self.time # T = t0 + 60min 

        # contadores
        Na = 0 # llegadas 

        i_llegada = [] # tiempos de llegada de la i-esima solicitud, ids son indices
        i_salida = [] # tiempos de salida de la i-esima solicitud, ids son indices
        player_TE = [] # Tiempos de cada jugador en espera
        queue = [] # cola de espera de todas las ruletas

        # eventos
        prox_llegada = self.next_ts(t) # tiempo de la proxima llegada
        spinTiempos = np.zeros(self.roulette_no) + np.infty # Se crea tiempos para cada ruleta
        tiempoEsperaRuletas = [] # Tiempo que cada persona espera desde su apuesta hasta que se gira la ruleta
        roulettees = [Roulette() for _ in range(self.roulette_no)] # Se crea una lista de ruletas
        while t < T: # Mientras no acceda el tiempo de cierre
            if prox_llegada <= min(spinTiempos):
                # si el proximo tiempo de llegada es antes del proximo tiempo de salida, se encola
                t = prox_llegada 
                Na += 1 # player_no, no empieza en 0
                prox_llegada = self.next_ts(t) # siguiente tiempo de llegada
                i_llegada.append(t)
                i_salida.append(np.infty)
                there_is_space = False
                for i in range(self.roulette_no):
                    if roulettees[i].check_space(): # Verifica disponibilidad en las ruletas
                        player_TE = np.append(player_TE,t - i_llegada[len(i_llegada)-1]) # Tiempo esperado por el jugador
                        # -------------------------------------------------------------------------------------------------
                        # POTENCIAL COLOCACIÓN DEL JUEGO Y SUS PROBS-------------------------------------------------------
                        spinTiempos[i] = t + np.random.exponential(45)
                        # POTENCIAL COLOCACIÓN DEL JUEGO Y SUS PROBS-------------------------------------------------------
                        # -------------------------------------------------------------------------------------------------
                        tiempoEsperaRuletas.append({ "player_no": Na, "roulette_no": i, "time": spinTiempos[i] - t })
                        roulettees[i].add_player_and_bet(Na, t)
                        there_is_space = True
                        break;
                # Si no hay espacio para el jugador, se encola
                if not there_is_space:
                    queue.append(Na)
            
            else:
                # si el proximo tiempo de llegada es después del próximo tiempo de salida, se atiende
                roulettePED = np.argmin(spinTiempos) # INDICE DE LA RULETA -------------------------------------------
                t = spinTiempos[roulettePED] # EXTRACCIÓN DE TIEMPO DE SALIDA-----------------------------------------
                roulettees[roulettePED].spin() # Se gira la ruleta
                spinTiempos[roulettePED] = np.infty # Agrega un valor infinito al tiempo de salida
                # Revisar si los jugadores se quedan en la ruleta
                players_keep, players_leave, spaces_availables = roulettees[roulettePED].reopen_roulette()
                # Se agregan las salidas de los jugadores que se van
                for player_no in players_leave:
                    i_salida[player_no - 1] = t
                # Se calcula el tiempo del proximo spin de esta ruleta
                spinTiempos[roulettePED] = t + np.random.exponential(45)
                # Se agregan las apuestas de los jugadores que se quedan
                for player_no in players_keep:
                    tiempoEsperaRuletas.append({ "player_no": player_no, "roulette_no": roulettePED, "time": spinTiempos[roulettePED] - t })
                    roulettees[roulettePED].set_bet(player_no, t)
                # Si hay jugadores en la cola, se agregan a la ruleta
                for _ in range(spaces_availables):
                    if len(queue) > 0:
                        player_no = queue.pop(0)
                        player_TE = np.append(player_TE,t - i_llegada[player_no - 1]) # Tiempo esperado por el jugador
                        # no se calcula el tiempo de spin, ya se calculó antes
                        tiempoEsperaRuletas.append({ "player_no": player_no, "roulette_no": roulettePED, "time": spinTiempos[roulettePED] - t })
                        roulettees[roulettePED].add_player_and_bet(player_no, t)
        
        minimal_arr = min(
            len(roulettees),
            len(i_llegada),
            len(i_salida)
                    )
        # -------------------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------------------
        # Una vez se tengan los datos de las rondas, dinero, % de winrate, saldo final, Rojos, Verdes y Negras
        # Se colocaran aqui. SE RECOMIENDA METERLOS EN UN ARRAY
        df = pd.DataFrame({
            'roulette_N':roulettees[0:minimal_arr],
            'arriving':i_llegada[0:minimal_arr],
            'leaving':i_salida[0:minimal_arr]
        })
        # print(df)

        # print("Indices de entrada: ",len(i_llegada))
        # print("Indices de salida: ",len(i_salida))  
        # print("Indices de salida: ",len(rouletteNo))     

        # se calcula cuantas Gamblers atendio cada ruleta 
        numGamblers = np.zeros(self.roulette_no)
        for roulette, index in zip(roulettees, range(self.roulette_no)):
            gamblersSet = set([bet["player_no"] for bet in roulette.bets])
            numGamblers[index] = len(gamblersSet)

        return { 
            "en_cola": player_TE,
            "numGamblers": numGamblers, 
            "setTiempo": spinTiempos, 
            "i_llegada": i_llegada, 
            "i_salida": i_salida, 
            "tiempoEsperaRuletas": tiempoEsperaRuletas
        }

def show_results(FdS):
    resultados_FdS = FdS.simulate()
    print(Fore.GREEN + ">>>>>>>  Casino La Gamba Sureña" + Style.RESET_ALL)
    print(Fore.GREEN + ">>>>>>>  1 Gambler cada 5 minutos" + Style.RESET_ALL)
    print(Fore.GREEN + ">>>>>>>  6 Ruletas en total" + Style.RESET_ALL)
    print(Fore.GREEN + ">>>>>>>  Simulación de 8 horas laborales" + Style.RESET_ALL + '\n')

    print("1. En promedio ¿cuánto tiempo estuvo cada player en cola?")
    print(np.round(np.mean(resultados_FdS["en_cola"]),5), "min")

    print("2. Número de player en la cola")
    print(resultados_FdS["numGamblers"], "min")

    clientsPerCashier = resultados_FdS["numGamblers"]
    totalClients = 0
    useIndex = []

    for cli in range(len(clientsPerCashier)):
        print('ruleta ', cli+1, ': ', str(clientsPerCashier[cli]), 'jugadores')
        totalClients += clientsPerCashier[cli]

    for i in range(len(clientsPerCashier)):
        useIndex.append(clientsPerCashier[i] / totalClients)

    print("3. Calcule el grado de utilización de cada ruleta")

    for i in range(len(useIndex)):
        print('Rendimiento de ruleta ', i+1, ': ', str(useIndex[i]))

    plt.bar(range(1, len(clientsPerCashier) + 1), useIndex)
    plt.title('Rendimiento de ruleta')
    plt.xlabel('ruleta')
    plt.ylabel('Porcentaje de Gamblers de ruleta')
    plt.show()

    # Grafica de tiempo de la simulacion
    len_min = min(len(resultados_FdS["i_llegada"]), len(resultados_FdS["i_salida"]))
    figure = px.scatter(x=resultados_FdS["i_llegada"][:len_min], y=resultados_FdS["i_salida"][:len_min], title="Tiempo de simulación", labels={"x": "Tiempo Llegada", "y": "Tiempo Salida"})
    figure.show()

    print(resultados_FdS["tiempoEsperaRuletas"][0], "min")
    print("\n3. ¿Cuánto tiempo estuvo cada roulette desocupado (idle)?")
    print(np.maximum(np.ones(FdS.roulette_no)*60 - resultados_FdS["tiempoEsperaRuletas"],0)[0], "min")
    print("\n4. Cuánto tiempo en total estuvieron las Gamblers en cola?")
    print(np.round(sum(resultados_FdS["en_cola"]),5), "min")

    print("\n6. En promedio, ¿cuántas Gamblers estuvieron en cola cada minuto?")
    sol_psec = [ 1/num if num != 0 else 0 for num in resultados_FdS["en_cola"] ]
    print(np.round(np.mean(sol_psec),5), "min")
    print("\n7. ¿Cuál es el momento de la salida de la última solicitud?")
    print(np.round(resultados_FdS["setTiempo"][-1],5), "min")

def show_mult_results(FdSs):
    # comparacion jugadores en cola
    clients_queue = [np.round(np.mean(result["en_cola"]),5) for result in FdSs]
    figure = px.bar(x=clients_queue, y=[f"{results['roulette']} roulette y {results['max_sol']} max_sol" for results in FdSs] , title="Tiempos promedio de jugadores en cola para cada simulación")
    figure.show()

def main():
    # Abre el casino Gamba Sureña
    # Promedio de clientes
    average_incidence = 0.25
    # 6 Ruletas en el casino
    roulette_n = 6
    # cantidad de horas de simulacion
    hours = 8
    FdS = Casino(average_incidence=average_incidence, roulette_no=roulette_n, time=hours)
    show_results(FdS)

    # # Resultados para diferentes valores de roulette
    # valores = [ { "roulette": 7, "max_sol": 4 }, { "roulette": 8, "max_sol": 5 },  { "roulette": 9, "max_sol": 6 },  { "roulette": 10, "max_sol": 7 } ]
    # mult_results = []
    # for val in valores:
    #     FdS = Casino(max_sol=val["max_sol"], roulette=val["roulette"])
    #     results = FdS.simulate()
    #     results["roulette"] = val["roulette"]
    #     results["max_sol"] = val["max_sol"]
    #     mult_results.append(results)
    # show_mult_results(mult_results)

if __name__ == "__main__":
    main()
