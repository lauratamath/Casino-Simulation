import numpy as np
import pandas as pd
import plotly.express as px
from matplotlib import pyplot as plt
from colorama import Fore, Style

class Casino():

    def __init__(self, max_sol = 0.25, roulette = 5, time = 8):
        self.lambda_max = max_sol  # cantidad máxima de Gamblers por mesa 
        self.roulette = roulette # cantidad de ruletas que tiene el casino
        self.time = time #cantidad de horas de simulacion
    # Calculo de siguiente llegada
    def next_ts(self, t): # No hace gran cosa, pero existe porque los eventos serán procesos de poisson homogeneos
        return t - (np.log(np.random.uniform())/self.lambda_max)
    # Calculo para tiempo
    # ----------------------MODIFICAR PARA PODER CAMBIAR LA CANTIDAD DE TIEMPO DE JUEGO---------------------------
    def get_exponential(self, lamda):
        return -(1 / lamda)*np.log(np.random.uniform())

    def simulate(self):
        t = 0 
        n = 0 # estado del sistema, numero de Gamblers en t
        T = 60*self.time # T = t0 + 60min 

        # contadores
        Na = 0 # llegadas 

        i_llegada = [] # tiempos de llegada de la i-esima solicitud, ids son indices
        i_salida = [] # tiempos de salida de la i-esima solicitud, ids son indices
        player_TE = [] # Tiempos de cada jugador en espera

        # eventos
        prox_llegada = self.next_ts(t) # tiempo de la proxima llegada
        setTiempo = np.zeros(self.roulette) + np.infty # Se crea tiempos en cero para cada ruleta
        tiempoOcupado = np.zeros(self.roulette) # Se crea tiempos iniciales que cada ruleta estuvo ocupado
        rouletteNo = [] # se guardan cuales Gamblers fueron atendidas por cual ruleta
        roulettees = np.zeros(self.roulette) # para llevar registro de cual está ocupado
        e = 0
        while t < T: # Mientras no acceda el tiempo de cierre
            if prox_llegada <= min(setTiempo):
                # si el proximo tiempo de llegada es antes del proximo tiempo de salida, se encola
                t = prox_llegada 
                Na += 1 
                prox_llegada = self.next_ts(t) # siguiente tiempo de llegada
                i_llegada.append(t)
                if n < self.roulette: # si hay menos jugadores dentro que roulette, se le asigna uno que esté disponible
                    for i in range(self.roulette):
                        if roulettees[i] == 0: # Verifica disponibilidad en las ruletas
                            player_TE = np.append(player_TE,t - i_llegada[len(i_llegada)-1]) # Tiempo esperado por el jugador
                            rouletteNo.append(i) # Registro de actividad en ruleta
                            # -------------------------------------------------------------------------------------------------
                            # POTENCIAL COLOCACIÓN DEL JUEGO Y SUS PROBS-------------------------------------------------------
                            setTiempo[i] = t + np.random.exponential(45)
                            # POTENCIAL COLOCACIÓN DEL JUEGO Y SUS PROBS-------------------------------------------------------
                            # -------------------------------------------------------------------------------------------------
                            tiempoOcupado[i] += setTiempo[i]-t 
                            roulettees[i] = 1 
                            break;
                n += 1 # Se agrega al nuevo player en el sistema
            
            else:
                # si el proximo tiempo de llegada es después del próximo tiempo de salida, se atiende 
                roulettePED = np.argmin(setTiempo) # INDICE DE LA RULETA -------------------------------------------
                t = setTiempo[roulettePED] # EXTRACCIÓN DE TIEMPO DE SALIDA-----------------------------------------
                i_salida.append(t) # Adhesion del registro de salida
                if n <= self.roulette: # Si hay menos o igual cantidad de jugadores que roulette
                    roulettees[roulettePED] = 0 # Coloca disponible la ruleta
                    setTiempo[roulettePED] = np.infty # Agrega un valor infimo al tiempo de salida
                else: # Hay mas jugadores por atender
                    rouletteNo.append(roulettePED) # Deja registro de la actividad de las ruletas
                    player_TE = np.append(player_TE,t - i_llegada[len(i_llegada)-1]) # Calculo de tiempo por ruleta
                    
                    # -------------------------------------------------------------------------------------------------
                    # POTENCIAL COLOCACIÓN DEL JUEGO Y SUS PROBS-------------------------------------------------------
                    setTiempo[i] = t + np.random.exponential(45) 
                    # POTENCIAL COLOCACIÓN DEL JUEGO Y SUS PROBS-------------------------------------------------------
                    # -------------------------------------------------------------------------------------------------

                    tiempoOcupado[roulettePED] += setTiempo[roulettePED]-t
                    roulettees[i] = 1 
                n -= 1 # Descontamos al player atendido del sistema
        
        minimal_arr = min(
            len(rouletteNo),
            len(i_llegada),
            len(i_salida)
                    )
        # -------------------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------------------
        # Una vez se tengan los datos de las rondas, dinero, % de winrate, saldo final, Rojos, Verdes y Negras
        # Se colocaran aqui. SE RECOMIENDA METERLOS EN UN ARRAY
        df = pd.DataFrame({
            'roulette_N':rouletteNo[0:minimal_arr],
            'arriving':i_llegada[0:minimal_arr],
            'leaving':i_salida[0:minimal_arr]
        })
        # print(df)

        # print("Indices de entrada: ",len(i_llegada))
        # print("Indices de salida: ",len(i_salida))  
        # print("Indices de salida: ",len(rouletteNo))     

        # se calcula cuantas Gamblers atendio cada ruleta 
        numGamblers = np.zeros(self.roulette)
        for i in range(len(rouletteNo)):
            numGamblers[rouletteNo[i]] += 1

        return { 
            "en_cola": player_TE,
            "numGamblers": numGamblers, 
            "setTiempo": setTiempo, 
            "i_llegada": i_llegada, 
            "i_salida": i_salida, 
            "tiempoOcupado": tiempoOcupado
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

    print(resultados_FdS["tiempoOcupado"][0], "min")
    print("\n3. ¿Cuánto tiempo estuvo cada roulette desocupado (idle)?")
    print(np.maximum(np.ones(FdS.roulette)*60 - resultados_FdS["tiempoOcupado"],0)[0], "min")
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
    counter = 6 # Numero de roulette

    # ## Pizzita Computing
    FdS = Casino(max_sol=0.2, roulette=counter)
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
