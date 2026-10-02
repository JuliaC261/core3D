#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fisica do nucleo - so a regiao ativa dos fuel rods (Wire Things).

Modelo didatico de um reator de agua pressurizada com os numeros de Angra 2
(projeto Siemens/KWU, 4 lacos, 193 elementos 16x16, 3771 MWt). Os calculos
rodam o tempo todo, a cada 100 ms, mesmo sem ninguem mexer em nada: xenonio,
calor de decaimento, mistura do boro e temperaturas continuam evoluindo.

O nucleo e dividido em nos: 21 colunas (uma por Wire Thing) x NOS_AXIAIS
fatias. Cada no tem a sua temperatura de pastilha, de revestimento e de agua,
a sua densidade de agua, o seu xenonio e o seu calor de decaimento. Os nos
conversam entre si de tres jeitos, como num reator real:

    1. Neutrons (difusao). O fluxo de um no depende da reatividade local
       (k-infinito) e do que vaza dos vizinhos. A forma do fluxo no nucleo
       inteiro e o modo fundamental da equacao de difusao, recalculado a
       cada passo; a reatividade do reator sai do autovalor. Um PWR grande
       e "frouxamente acoplado": pequenas diferencas locais mudam bastante
       a distribuicao, e e o realimentamento local (Doppler e moderador) que
       segura o nucleo equilibrado.
    2. Agua (mistura lateral). Os elementos de PWR nao tem caixa: a agua de
       um canal troca com a dos vizinhos enquanto sobe.
    3. Pleno inferior. Cada um dos 4 lacos entra por um bocal e alimenta
       principalmente o seu quadrante, mas se mistura em parte com os outros
       antes de subir pelo nucleo.

Quadrantes: Q1 a Q4, vistos de cima (Q1 em cima a direita, sentido
anti-horario). O laco N alimenta o quadrante N. O QPTR (razao de inclinacao
de potencia entre quadrantes) e calculado como nas salas de controle.

Inicio
------
Ao abrir, o reator ja comeca conforme MODO_INICIAL, com a fisica rodando:
    "desligado" - (padrao) todas as barras no fundo, reator subcritico e
                  quente (agua a T_ENTRADA, 15,7 MPa), boro em BORO_INICIAL,
                  sem xenonio. So a fonte de partida mantem alguns neutrons.
                  O usuario faz a partida retirando os bancos.
    "partida"   - comeca desligado e o programa faz a partida sozinho:
                  retira SA e SB, depois CA e CB controlando a taxa de subida
                  (SUR), ate POTENCIA_ALVO_DA_PARTIDA. Qualquer botao do painel
                  ou um trip devolve o controle ao usuario.
    "operando"  - critico a POTENCIA_INICIAL, com xenonio de equilibrio e o
                  boro ja ajustado (busca do boro critico).
A tecla I reinicia a simulacao no MODO_INICIAL (tambem serve depois de um trip).

O que da para fazer para ver os quadrantes se influenciando:
    - teclas 1 a 4: escolhe o quadrante
    - D: derruba (ou realinha) uma barra do banco CA naquele quadrante
    - , e . : esfria / esquenta a perna fria do laco daquele quadrante
      (como se o gerador de vapor daquele laco tirasse mais ou menos calor)

Dados de Angra 2 usados (FSAR, via Gonzalez-Mantecon et al., EPJ Nuclear
Sci. Technol. 1, 5 (2015)): 3771,4 MWt; 193 elementos com 236 varetas;
revestimento de Zircaloy-4 com 0,72 mm; 15,7 MPa; temperatura media de
308,6 C com 34 C de elevacao no nucleo; vazao efetiva de 17 672 kg/s.
Altura ativa de 3,9 m, como nas usinas KWU de 16x16.

Os coeficientes de reatividade (Doppler, moderador, boro, xenonio), a mistura
entre canais e a do pleno inferior sao valores tipicos de PWR, nao medidos
em Angra 2. E um modelo para aprender como as coisas se relacionam, nao um
codigo de analise de seguranca.

Uso
---
Dentro do visualizador (o visualizador.py ja faz isso):

    from nucleo import MonitorDoNucleo
    nucleo = MonitorDoNucleo(menu.partes, barras, painel, janela, renderizador)
    nucleo.conectar(interator)

Teclas: J diluir boro, K borar, X acelerar a fisica, 1-4 quadrante,
        D barra caida, , e . temperatura da perna fria, I reiniciar.

Sozinho, sem VTK, roda testes no terminal:

    python nucleo.py

Requer numpy (ja vem junto com o vtk pelo pip).
"""

from __future__ import annotations

import math
import time

import numpy as np

# ----------------------------------------------------------------------------
# Condicoes de contorno (o que esta fora da regiao dos fuel rods)
# ----------------------------------------------------------------------------
PRESSAO_MPA = 15.7                   # pressao do primario (pressurizador)
T_ENTRADA = 291.6                    # perna fria, em C (308,6 - 34/2)
VAZAO_NUCLEO = 17672.0               # kg/s efetivos para troca de calor
POTENCIA_NOMINAL = 3771.4e6          # W termicos a 100 %
FRACAO_NO_COMBUSTIVEL = 0.974        # o resto (gama, neutrons) aquece a agua direto

# ----------------------------------------------------------------------------
# Geometria de um elemento 16x16 de Angra 2 (o modelo 3D da so a forma)
# ----------------------------------------------------------------------------
ELEMENTOS_REAIS = 193
VARETAS_POR_ELEMENTO = 236           # 16x16 menos 20 tubos-guia
LARGURA_DO_ELEMENTO = 0.23           # m, de centro a centro
ALTURA_ATIVA = 3.90                  # m
RAIO_PASTILHA = 4.555e-3             # m (pastilha de 9,11 mm)
RAIO_EXTERNO_REVESTIMENTO = 5.375e-3 # vareta de 10,75 mm
RAIO_INTERNO_REVESTIMENTO = 5.375e-3 - 0.72e-3
PASSO_DAS_VARETAS = 14.3e-3          # m, distancia entre centros
COEF_GAP = 5700.0                    # W/m2K, condutancia do gap (He)
K_REVESTIMENTO = 17.0                # W/mK, Zircaloy-4
RHO_CP_UO2 = 3.2e6                   # J/m3K, capacidade termica da pastilha

NOS_AXIAIS = 10                      # fatias do combustivel na vertical

# ----------------------------------------------------------------------------
# Difusao de neutrons (um grupo, malha grossa)
# ----------------------------------------------------------------------------
# AREA_DE_MIGRACAO e efetiva, nao o M^2 fisico da agua (~60 cm2): cada no
# do modelo representa ~9 elementos, e o valor foi ajustado para o nucleo ter
# o acoplamento de um PWR grande (razao de dominancia ~0,95) e bancos com
# valores coerentes com a sequencia de partida.
AREA_DE_MIGRACAO = 0.030             # m2
ECONOMIA_DO_REFLETOR = 0.08          # m, o refletor "estica" o nucleo
CARREGAMENTO_PERIFERICO = 0.10       # combustivel novo na borda achata o fluxo
DESLOCAMENTO_WIELANDT = 0.001        # acelera o calculo do modo fundamental

# ----------------------------------------------------------------------------
# Mistura da agua
# ----------------------------------------------------------------------------
MISTURA_LATERAL = 0.02               # fracao trocada com cada vizinho por fatia
MISTURA_DO_PLENO = (0.70, 0.12, 0.06) # proprio quadrante, vizinho, oposto

# ----------------------------------------------------------------------------
# Cinetica (U-235) e fonte de partida
# ----------------------------------------------------------------------------
BETA_GRUPOS = (0.000215, 0.001424, 0.001274, 0.002568, 0.000748, 0.000273)
LAMBDA_GRUPOS = (0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01)   # 1/s
BETA = sum(BETA_GRUPOS)
TEMPO_DE_GERACAO = 2.0e-5            # s (Lambda)
FONTE = 3.0e-5                       # fonte de neutrons, em fracao nominal/s
PASSO_CINETICA = 0.01                # s

# ----------------------------------------------------------------------------
# Reatividade (pcm = 1e-5)
#
# Referencia: zero de potencia quente (HZP), agua a T_ENTRADA, sem xenonio,
# barras fora. RHO_EXCESSO e a reatividade do combustivel nessa condicao,
# sem boro. Com o boro inicial sobra uns +2000 pcm, que as barras seguram.
# Cada termo abaixo vale no no onde acontece; o total sai da difusao.
# ----------------------------------------------------------------------------
RHO_EXCESSO = 11600.0
BORO_INICIAL = 1200.0                # ppm
VALOR_DO_BORO = 8.0                  # pcm por ppm, na densidade de referencia
COEF_MODERACAO = 14300.0             # pcm por unidade de densidade relativa
COEF_DOPPLER = -150.0                # pcm por raiz(K) da temperatura efetiva
VALOR_TOTAL_DAS_BARRAS = 8500.0      # pcm com todas as barras no fundo
XENONIO_EQUILIBRIO = 2800.0          # pcm de Xe em equilibrio a 100 %
TEMPO_DE_MISTURA_DO_BORO = 30.0      # s, para a concentracao chegar no alvo
PASSO_DO_BORO = 10.0                 # ppm por toque em J ou K
PASSO_DA_PERNA_FRIA = 1.0            # C por toque em , ou .

# Xenonio e iodo
LAMBDA_IODO = 2.87e-5                # 1/s
LAMBDA_XENONIO = 2.09e-5
RENDIMENTO_IODO = 0.0639
RENDIMENTO_XENONIO = 0.00237
QUEIMA_DO_XENONIO = 7.5e-5           # sigma_a * fluxo a 100 %, em 1/s

# Calor de decaimento: (fracao da potencia, constante de tempo em s)
GRUPOS_DE_DECAIMENTO = ((0.020, 3.0), (0.020, 40.0), (0.015, 400.0), (0.010, 4000.0))

# ----------------------------------------------------------------------------
# Inicio da simulacao
# ----------------------------------------------------------------------------
MODO_INICIAL = "desligado"           # "desligado", "partida" ou "operando"
POTENCIA_INICIAL = 1.0               # modo "operando": fracao nominal
XENONIO_DE_EQUILIBRIO = True         # modo "operando": Xe como apos dias a essa potencia
POSICOES_DE_OPERACAO = {"SA": 231, "SB": 231, "CA": 231, "CB": 200}   # passos
POTENCIA_ALVO_DA_PARTIDA = 1.0       # modo "partida": onde o programa para
SUR_MAXIMO_DA_PARTIDA = 1.0          # dpm; acima disso a partida espera

# ----------------------------------------------------------------------------
# Protecao e interface
# ----------------------------------------------------------------------------
TRIP_AUTOMATICO = True
LIMITE_DE_POTENCIA = 1.09            # fracao nominal (109 %)
TRIP_NA_SATURACAO = True             # agua do canal chegou a saturacao
LIMITE_QPTR = 1.02                   # alarme de inclinacao entre quadrantes
T_FUSAO_UO2 = 2840.0                 # C, so para o alarme

ACELERACOES = (1, 10, 60, 600)       # tecla X alterna entre essas
INTERVALO_DA_FISICA = 100            # ms entre atualizacoes na tela
ORCAMENTO_POR_QUADRO = 0.06          # s de CPU no maximo por atualizacao
PREFIXO_FUEL_RODS = "wire things"
BANCO_DA_BARRA_CAIDA = "CA"          # banco de onde sai a barra que cai (tecla D)
DIRECAO_VERTICAL = (0.0, 1.0, 0.0)   # mesmo "para cima" do visualizador

# Quadrantes vistos de cima: (sinal em x, sinal em z). Q1 em cima a direita.
SINAIS_DOS_QUADRANTES = ((1, -1), (-1, -1), (-1, 1), (1, 1))


# ----------------------------------------------------------------------------
# Agua a 15,7 MPa (IAPWS-IF97)
# colunas: T [C], densidade [kg/m3], entalpia [kJ/kg], cp [kJ/kgK],
#          condutividade [W/mK], viscosidade [uPa.s]
# ----------------------------------------------------------------------------
TABELA_AGUA = (
    (20.0, 1005.23, 98.59, 4.139, 0.6069, 997.36),
    (40.0, 998.94, 181.39, 4.142, 0.6366, 654.84),
    (60.0, 989.90, 264.30, 4.150, 0.6590, 469.82),
    (80.0, 978.65, 347.41, 4.163, 0.6753, 358.23),
    (100.0, 965.51, 430.85, 4.182, 0.6860, 285.77),
    (125.0, 946.71, 535.81, 4.217, 0.6922, 226.12),
    (150.0, 925.40, 641.78, 4.264, 0.6912, 186.44),
    (175.0, 901.57, 749.15, 4.329, 0.6843, 158.60),
    (200.0, 875.00, 858.43, 4.419, 0.6716, 138.10),
    (220.0, 851.52, 947.73, 4.515, 0.6573, 125.18),
    (240.0, 825.70, 1039.26, 4.644, 0.6396, 114.36),
    (250.0, 811.75, 1086.10, 4.725, 0.6295, 109.53),
    (260.0, 797.00, 1133.81, 4.820, 0.6184, 104.99),
    (270.0, 781.32, 1182.56, 4.933, 0.6065, 100.69),
    (275.0, 773.09, 1207.38, 4.998, 0.6001, 98.61),
    (280.0, 764.57, 1232.55, 5.069, 0.5935, 96.56),
    (285.0, 755.73, 1258.09, 5.149, 0.5866, 94.55),
    (290.0, 746.53, 1284.05, 5.238, 0.5795, 92.55),
    (295.0, 736.93, 1310.49, 5.338, 0.5721, 90.57),
    (300.0, 726.89, 1337.45, 5.451, 0.5643, 88.60),
    (305.0, 716.35, 1365.03, 5.582, 0.5562, 86.62),
    (310.0, 705.22, 1393.31, 5.734, 0.5478, 84.63),
    (315.0, 693.40, 1422.41, 5.913, 0.5389, 82.61),
    (320.0, 680.77, 1452.50, 6.129, 0.5295, 80.54),
    (325.0, 667.14, 1483.78, 6.395, 0.5196, 78.41),
    (330.0, 652.28, 1516.57, 6.732, 0.5091, 76.18),
    (335.0, 635.80, 1551.29, 7.182, 0.4977, 73.81),
    (338.0, 624.90, 1573.35, 7.539, 0.4905, 72.30),
    (341.0, 612.97, 1596.63, 8.002, 0.4829, 70.69),
    (344.0, 599.66, 1621.54, 8.638, 0.4748, 68.93),
    (345.83, 590.63, 1637.76, 9.159, 0.4697, 67.76),
)
T_SATURACAO = 345.83                 # C a 15,7 MPa
DENSIDADE_VAPOR = 104.09             # kg/m3, vapor saturado
ENTALPIA_LIQUIDO = 1637.76e3         # J/kg, liquido saturado
ENTALPIA_VAPOR = 2590.15e3           # J/kg, vapor saturado

_TAB = np.array(TABELA_AGUA)
_T, _RHO, _H = _TAB[:, 0], _TAB[:, 1], _TAB[:, 2] * 1e3
_CP, _K, _MU = _TAB[:, 3] * 1e3, _TAB[:, 4], _TAB[:, 5] * 1e-6


def entalpia_da_agua(temperatura):
    return np.interp(temperatura, _T, _H)


def estado_da_agua(entalpia):
    """(temperatura, densidade, titulo) a partir da entalpia em J/kg.

    Abaixo da saturacao e liquido comprimido. Acima, mistura homogenea de
    liquido e vapor saturados na temperatura de saturacao.
    """
    entalpia = np.asarray(entalpia, dtype=float)
    liquido = entalpia < ENTALPIA_LIQUIDO
    T = np.interp(entalpia, _H, _T)
    titulo = np.clip((entalpia - ENTALPIA_LIQUIDO) / (ENTALPIA_VAPOR - ENTALPIA_LIQUIDO), 0.0, 1.0)
    mistura = 1.0 / (titulo / DENSIDADE_VAPOR + (1.0 - titulo) / _RHO[-1])
    densidade = np.where(liquido, np.interp(T, _T, _RHO), mistura)
    return np.where(liquido, T, T_SATURACAO), densidade, titulo


DENSIDADE_REFERENCIA = float(np.interp(T_ENTRADA, _T, _RHO))


# ----------------------------------------------------------------------------
# Pequenas funcoes de fisica
# ----------------------------------------------------------------------------
def condutividade_uo2(temperatura_c):
    """Condutividade do UO2 (W/mK): cai com a temperatura ate uns 1800 C."""
    T = np.maximum(temperatura_c, 20.0) + 273.15
    return 1.0 / (0.0375 + 2.165e-4 * T) + 4.715e9 / T ** 2 * np.exp(-16361.0 / T)


def superaquecimento_jens_lottes(fluxo):
    """Quanto o revestimento fica acima da saturacao em ebulicao nucleada."""
    return 25.0 * (np.maximum(fluxo, 0.0) / 1e6) ** 0.25 * math.exp(-PRESSAO_MPA / 6.2)


# ----------------------------------------------------------------------------
# O nucleo (sem VTK: da para testar sozinho)
# ----------------------------------------------------------------------------
class NucleoPWR:
    """Estado e evolucao do nucleo, no a no, so na regiao ativa do combustivel.

    `posicoes` sao os centros de cada coluna de combustivel em passos da
    grade (0, 0 e o centro; +x a direita e -z para cima, vistos de cima).
    `insercoes` e, por coluna, a fracao da altura ativa ocupada pela barra,
    contada de cima (0 = fora, 1 = no fundo).
    """

    def __init__(self, posicoes: list, insercoes: list | None = None,
                 nos: int = NOS_AXIAIS) -> None:
        C, N = len(posicoes), nos
        self.colunas, self.nos = C, N
        self.posicoes = np.array(posicoes, dtype=float)
        forma = (C, N)

        # -- vizinhanca radial ----------------------------------------------
        chave = {(round(x), round(z)): j for j, (x, z) in enumerate(posicoes)}
        self.vizinhos = [
            [chave.get((round(x) + dx, round(z) + dz)) for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1))]
            for x, z in posicoes
        ]
        self.laplaciano = np.zeros((C, C))
        for j, viz in enumerate(self.vizinhos):
            for v in viz:
                if v is not None:
                    self.laplaciano[j, j] += 1.0
                    self.laplaciano[j, v] -= 1.0

        # -- quadrantes: peso de cada coluna em cada quadrante ---------------
        self.pesos_quadrante = np.zeros((4, C))
        for q, (sx, sz) in enumerate(SINAIS_DOS_QUADRANTES):
            for j, (x, z) in enumerate(self.posicoes):
                fx = 0.5 if abs(x) < 0.5 else (1.0 if x * sx > 0 else 0.0)
                fz = 0.5 if abs(z) < 0.5 else (1.0 if z * sz > 0 else 0.0)
                self.pesos_quadrante[q, j] = fx * fz
        proprio, vizinho, oposto = MISTURA_DO_PLENO
        self.mistura_do_pleno = np.array([
            [proprio if a == b else (oposto if (a - b) % 4 == 2 else vizinho) for b in range(4)]
            for a in range(4)
        ])

        # -- difusao: matriz de fuga (constante) -----------------------------
        # Cada coluna do modelo vale ELEMENTOS_REAIS/C elementos: o no e mais
        # largo que um elemento, com a mesma area total do nucleo real.
        largura = LARGURA_DO_ELEMENTO * math.sqrt(ELEMENTOS_REAIS / C)
        self.dz = ALTURA_ATIVA / N
        M2, d = AREA_DE_MIGRACAO, ECONOMIA_DO_REFLETOR
        a_r, a_z = M2 / largura ** 2, M2 / self.dz ** 2
        a_rb = M2 / (largura * (largura / 2.0 + d))
        a_zb = M2 / (self.dz * (self.dz / 2.0 + d))
        A = np.eye(C * N)
        for j in range(C):
            for k in range(N):
                i = j * N + k
                for v in self.vizinhos[j]:
                    if v is None:
                        A[i, i] += a_rb
                    else:
                        A[i, i] += a_r
                        A[i, v * N + k] -= a_r
                for kk in (k - 1, k + 1):
                    if 0 <= kk < N:
                        A[i, i] += a_z
                        A[i, j * N + kk] -= a_z
                    else:
                        A[i, i] += a_zb
        self.A = A

        raio = np.hypot(self.posicoes[:, 0], self.posicoes[:, 1])
        self.carregamento = np.repeat(
            (1.0 + CARREGAMENTO_PERIFERICO * (raio / max(raio.max(), 1e-9)) ** 2)[:, None], N, axis=1)

        # -- canal termico: um canal do modelo = ELEMENTOS_REAIS/C elementos -
        self.varetas = ELEMENTOS_REAIS * VARETAS_POR_ELEMENTO / C
        self.vazao = VAZAO_NUCLEO / C
        area_por_vareta = PASSO_DAS_VARETAS ** 2 - math.pi * RAIO_EXTERNO_REVESTIMENTO ** 2
        self.fluxo_massico = self.vazao / (self.varetas * area_por_vareta)
        self.diametro_hidraulico = 4.0 * area_por_vareta / (2.0 * math.pi * RAIO_EXTERNO_REVESTIMENTO)
        self.capacidade = RHO_CP_UO2 * math.pi * RAIO_PASTILHA ** 2   # J/mK
        self.r_gap = 1.0 / (2.0 * math.pi * RAIO_PASTILHA * COEF_GAP)
        self.r_revest = math.log(RAIO_EXTERNO_REVESTIMENTO / RAIO_INTERNO_REVESTIMENTO) / (
            2.0 * math.pi * K_REVESTIMENTO)

        # -- estado termico -------------------------------------------------
        self.T_pastilha = np.full(forma, T_ENTRADA)
        self.T_centro = np.full(forma, T_ENTRADA)
        self.T_efetiva = np.full(forma, T_ENTRADA)
        self.T_revestimento = np.full(forma, T_ENTRADA)
        self.T_agua = np.full(forma, T_ENTRADA)
        self.densidade = np.full(forma, DENSIDADE_REFERENCIA)
        self.titulo = np.zeros(forma)
        self.q_linear = np.zeros(forma)          # W/m gerado por vareta
        self.q_saida = np.zeros(forma)           # W/m entregue a agua
        self.T_saida = np.full(C, T_ENTRADA)
        self.entalpia_saida = np.full(C, float(entalpia_da_agua(T_ENTRADA)))
        self.perna_fria = np.full(4, T_ENTRADA)  # um valor por laco

        # -- barras, boro, xenonio, decaimento ------------------------------
        self.insercao = np.ones(C) if insercoes is None else np.clip(np.array(insercoes, float), 0, 1)
        self.coberto = np.zeros(forma)
        self._cobertura()
        self.boro = BORO_INICIAL
        self.boro_alvo = BORO_INICIAL
        self.iodo = np.zeros(forma)
        self.xenonio = np.zeros(forma)
        self.xenonio_100 = (RENDIMENTO_IODO + RENDIMENTO_XENONIO) / (
            LAMBDA_XENONIO + QUEIMA_DO_XENONIO)
        self.decaimento = np.zeros((len(GRUPOS_DE_DECAIMENTO),) + forma)
        self.calor = np.zeros(forma)             # potencia termica por no (fracao nominal)
        self.passo_termico = 0.05
        self.tempo = 0.0
        self.sur = 0.0                           # decadas por minuto
        self.pico_de_neutrons = 0.0

        # -- calibracao: K0 da o excesso; R da o valor total das barras ------
        self.K0, self.R = 1.0, 0.0
        self.phi = np.ones(C * N)
        k_ref = self._autovalor_exato(self.carregamento)
        k_alvo = 1.0 / (1.0 - RHO_EXCESSO * 1e-5)
        k_dentro = 1.0 / (1.0 - (RHO_EXCESSO - VALOR_TOTAL_DAS_BARRAS) * 1e-5)
        self.K0 = k_alvo / k_ref
        self.R = 1.0 - k_dentro / k_alvo         # todas dentro = k cai por igual

        # -- estado inicial: subcritico com fonte ---------------------------
        self._reatividade(exato=True)
        rho = self.rho_total * 1e-5
        self.n = FONTE * TEMPO_DE_GERACAO / -rho if rho < -1e-4 else 1e-6
        self.precursores = [
            b * self.n / (l * TEMPO_DE_GERACAO) for b, l in zip(BETA_GRUPOS, LAMBDA_GRUPOS)
        ]
        self._distribuir(0.0)

    # -- entradas --------------------------------------------------------------
    def definir_insercoes(self, insercoes) -> None:
        self.insercao = np.clip(np.array(insercoes, dtype=float), 0.0, 1.0)
        self._cobertura()

    def mudar_boro(self, delta_ppm: float) -> None:
        self.boro_alvo = max(0.0, min(3000.0, self.boro_alvo + delta_ppm))

    def mudar_perna_fria(self, laco: int, delta: float) -> None:
        novo = self.perna_fria[laco] + delta
        self.perna_fria[laco] = min(T_ENTRADA + 30.0, max(T_ENTRADA - 30.0, novo))

    def equilibrar(self, potencia: float, xenonio: bool = True,
                   procurar_boro: bool = True) -> bool:
        """Poe o nucleo em regime permanente na potencia pedida.

        Temperaturas, calor de decaimento, precursores e (se pedido) xenonio
        ficam no equilibrio daquela potencia, com as barras onde estao. Com
        procurar_boro, ajusta o boro ate o reator ficar critico, como a
        "busca do boro critico" da usina. Devolve True se conseguiu.
        """
        imediata = 1.0 - sum(f for f, _ in GRUPOS_DE_DECAIMENTO)
        self.n = max(potencia, 1e-9)
        for iteracao in range(100):
            fissao = self.n * self.forma_de_potencia
            for g, (fracao, _) in enumerate(GRUPOS_DE_DECAIMENTO):
                self.decaimento[g] = fracao * fissao
            self.calor = fissao * imediata + self.decaimento.sum(axis=0)
            if xenonio:
                self.iodo = RENDIMENTO_IODO * fissao / LAMBDA_IODO
                self.xenonio = (RENDIMENTO_IODO + RENDIMENTO_XENONIO) * fissao / (
                    LAMBDA_XENONIO + QUEIMA_DO_XENONIO * fissao)
            else:
                self.iodo = np.zeros_like(fissao)
                self.xenonio = np.zeros_like(fissao)
            self._termica(1e5)
            self._termica(1e5)
            self._reatividade(exato=True)
            if not procurar_boro:
                if iteracao >= 10:
                    break
                continue
            por_ppm = (-self.parcelas["Boro"] / self.boro if self.boro > 1.0
                       else VALOR_DO_BORO * float(self.densidade.mean()) / DENSIDADE_REFERENCIA)
            novo = min(3000.0, max(0.0, self.boro + self.rho_total / por_ppm))
            if abs(self.rho_total) < 0.3 and abs(novo - self.boro) < 0.05:
                break
            self.boro = novo
        self.boro_alvo = self.boro
        self.precursores = [
            b * self.n / (l * TEMPO_DE_GERACAO) for b, l in zip(BETA_GRUPOS, LAMBDA_GRUPOS)
        ]
        self.sur = 0.0
        self.pico_de_neutrons = self.n
        return abs(self.rho_total) < 5.0

    def _cobertura(self) -> None:
        """Fracao de cada no (coluna, fatia) que a barra ocupa."""
        base = np.arange(self.nos) / self.nos
        topo = base + 1.0 / self.nos
        inicio = (1.0 - self.insercao)[:, None]
        self.coberto = np.clip(topo[None, :] - np.maximum(base[None, :], inicio), 0.0, None) * self.nos

    # -- saidas ----------------------------------------------------------------
    @property
    def potencia_termica(self) -> float:
        return float(self.calor.mean())

    @property
    def keff(self) -> float:
        return self.k_eig

    # -- difusao ---------------------------------------------------------------
    def _autovalor_exato(self, kinf) -> float:
        """Modo fundamental direto (problema simetrico generalizado)."""
        F = kinf.ravel()
        raiz = 1.0 / np.sqrt(F)
        S = raiz[:, None] * self.A * raiz[None, :]
        valores, vetores = np.linalg.eigh(S)
        phi = np.abs(vetores[:, 0]) * raiz
        self.phi = phi / phi.mean()
        self.k_eig = 1.0 / valores[0]
        return self.k_eig

    def _autovalor(self, kinf, iteracoes: int = 3) -> float:
        """Iteracao inversa com deslocamento de Wielandt, partindo da forma anterior."""
        F = kinf.ravel()
        phi = self.phi
        lam = (phi @ self.A @ phi) / (phi @ (F * phi))
        inversa = np.linalg.inv(self.A - (lam - DESLOCAMENTO_WIELANDT) * np.diag(F))
        for _ in range(iteracoes):
            phi = inversa @ (F * phi)
            phi /= phi.mean()
        if phi.min() <= 0.0:                     # mudanca grande demais: resolve direto
            return self._autovalor_exato(kinf)
        self.phi = phi
        self.k_eig = float((phi @ (F * phi)) / (phi @ self.A @ phi))
        return self.k_eig

    # -- evolucao --------------------------------------------------------------
    def avancar(self, segundos: float, limite_de_cpu: float | None = None) -> float:
        """Avanca o tempo. Devolve quanto tempo de fisica foi simulado."""
        inicio = time.perf_counter()
        feito = 0.0
        self.pico_de_neutrons = self.n           # a protecao olha o maximo do trecho
        while feito < segundos - 1e-9:
            rho = self.rho_total * 1e-5
            # Perto do pronto-critico o realimentamento precisa ser fino.
            teto = 0.002 if rho > 0.8 * BETA else self.passo_termico
            h = min(segundos - feito, teto)
            n_antes = self.n
            self._cinetica(h, rho)
            self.pico_de_neutrons = max(self.pico_de_neutrons, self.n)
            self._distribuir(h)
            self._termica(h)
            self._reatividade()
            if n_antes > 0.0 and self.n > 0.0:
                instantaneo = 26.06 * math.log(self.n / n_antes) / h
                self.sur += (instantaneo - self.sur) * (1.0 - math.exp(-h / 1.0))
            feito += h
            self.tempo += h
            if limite_de_cpu is not None and time.perf_counter() - inicio > limite_de_cpu:
                break
        return feito

    def _cinetica(self, h: float, rho: float) -> None:
        """Euler implicito da cinetica pontual (estavel mesmo com passo grande)."""
        passo = PASSO_CINETICA
        if rho > BETA:
            passo = min(passo, 0.5 * TEMPO_DE_GERACAO / (rho - BETA))
        restante = h
        while restante > 1e-12:
            dt = min(passo, restante)
            fatores = [1.0 / (1.0 + dt * l) for l in LAMBDA_GRUPOS]
            numerador = self.n + dt * FONTE + dt * sum(
                l * c * f for l, c, f in zip(LAMBDA_GRUPOS, self.precursores, fatores))
            denominador = 1.0 - dt * (rho - BETA) / TEMPO_DE_GERACAO - sum(
                dt * l * dt * b / TEMPO_DE_GERACAO * f
                for l, b, f in zip(LAMBDA_GRUPOS, BETA_GRUPOS, fatores))
            self.n = min(1e4, max(0.0, numerador / denominador))
            self.precursores = [
                (c + dt * b / TEMPO_DE_GERACAO * self.n) * f
                for c, b, f in zip(self.precursores, BETA_GRUPOS, fatores)
            ]
            restante -= dt

    def _distribuir(self, h: float) -> None:
        """Fissao por no, boro, xenonio/iodo e calor de decaimento locais."""
        self.boro = self.boro_alvo + (self.boro - self.boro_alvo) * math.exp(
            -h / TEMPO_DE_MISTURA_DO_BORO)

        fissao = self.n * self.forma_de_potencia          # media = n
        self.iodo += h * (RENDIMENTO_IODO * fissao - LAMBDA_IODO * self.iodo)
        self.xenonio += h * (RENDIMENTO_XENONIO * fissao + LAMBDA_IODO * self.iodo
                             - LAMBDA_XENONIO * self.xenonio
                             - QUEIMA_DO_XENONIO * fissao * self.xenonio)
        np.maximum(self.xenonio, 0.0, out=self.xenonio)

        imediata = 1.0 - sum(f for f, _ in GRUPOS_DE_DECAIMENTO)
        for g, (fracao, tau) in enumerate(GRUPOS_DE_DECAIMENTO):
            alvo = fracao * fissao
            self.decaimento[g] = alvo + (self.decaimento[g] - alvo) * math.exp(-h / tau)
        self.calor = fissao * imediata + self.decaimento.sum(axis=0)

    def _termica(self, h: float) -> None:
        """Pastilha -> gap -> revestimento -> pelicula -> agua, fatia a fatia."""
        por_no = self.varetas * self.dz
        q_no = POTENCIA_NOMINAL * self.calor / (self.colunas * self.nos) / por_no
        re_ro = 2.0 * math.pi * RAIO_EXTERNO_REVESTIMENTO

        # Pleno inferior: cada coluna recebe a mistura dos lacos do(s) seu(s) quadrante(s).
        h_lacos = entalpia_da_agua(self.perna_fria)
        entalpia = self.pesos_quadrante.T @ (self.mistura_do_pleno @ h_lacos)

        for k in range(self.nos):
            q_comb = q_no[:, k] * FRACAO_NO_COMBUSTIVEL
            q_direto = q_no[:, k] - q_comb               # gama e neutrons na agua

            meio = entalpia + 0.5 * (self.q_saida[:, k] + q_direto) * por_no / self.vazao
            T_agua, densidade, titulo = estado_da_agua(meio)

            mu = np.interp(T_agua, _T, _MU)
            cp = np.interp(T_agua, _T, _CP)
            kw = np.interp(T_agua, _T, _K)
            reynolds = self.fluxo_massico * self.diametro_hidraulico / mu
            prandtl = cp * mu / kw
            pelicula = 0.023 * reynolds ** 0.8 * prandtl ** 0.4 * kw / self.diametro_hidraulico

            r_pastilha = 1.0 / (8.0 * math.pi * condutividade_uo2(self.T_pastilha[:, k]))
            r_fora = self.r_gap + self.r_revest + 1.0 / (re_ro * pelicula)
            r_total = r_pastilha + r_fora

            # Solucao exata do "capacitor" termico: estavel com passo grande.
            equilibrio = T_agua + q_comb * r_total
            tau = self.capacidade * r_total
            T_media = equilibrio + (self.T_pastilha[:, k] - equilibrio) * np.exp(-h / tau)
            q_saida = (T_media - T_agua) / r_total

            T_superficie = T_agua + q_saida * r_fora
            T_centro = 2.0 * T_media - T_superficie       # perfil parabolico
            T_parede = T_agua + q_saida / (re_ro * pelicula)
            teto = T_SATURACAO + superaquecimento_jens_lottes(q_saida / re_ro)
            T_parede = np.where(T_parede > T_SATURACAO,
                                np.maximum(T_agua, np.minimum(T_parede, teto)), T_parede)

            self.T_pastilha[:, k] = T_media
            self.T_centro[:, k] = T_centro
            self.T_efetiva[:, k] = T_superficie + 4.0 / 9.0 * (T_centro - T_superficie)
            self.T_revestimento[:, k] = T_parede
            self.T_agua[:, k] = T_agua
            self.densidade[:, k] = densidade
            self.titulo[:, k] = titulo
            self.q_linear[:, k] = q_no[:, k]
            self.q_saida[:, k] = q_saida

            entalpia = entalpia + (q_saida + q_direto) * por_no / self.vazao
            # Mistura com os canais vizinhos (elementos sem caixa).
            entalpia = entalpia - MISTURA_LATERAL * (self.laplaciano @ entalpia)

        self.entalpia_saida = entalpia
        self.T_saida = estado_da_agua(entalpia)[0]

    def _reatividade(self, exato: bool = False) -> None:
        rel = self.densidade / DENSIDADE_REFERENCIA
        termos = {
            "Boro": -VALOR_DO_BORO * self.boro * rel,
            "Moderador": COEF_MODERACAO * (rel - 1.0),
            "Doppler": COEF_DOPPLER * (np.sqrt(self.T_efetiva + 273.15) - math.sqrt(T_ENTRADA + 273.15)),
            "Xenônio": -XENONIO_EQUILIBRIO * self.xenonio / self.xenonio_100,
        }
        soma = sum(termos.values())
        kinf = self.K0 * self.carregamento * (1.0 + 1e-5 * soma) * (1.0 - self.R * self.coberto)
        if exato:
            self._autovalor_exato(kinf)
        else:
            self._autovalor(kinf)

        phi = self.phi.reshape(kinf.shape)
        potencia = self.carregamento * phi
        self.forma_de_potencia = potencia / potencia.mean()
        self.rho_total = (1.0 - 1.0 / self.k_eig) * 1e5

        # Parcelas pesadas pela importancia (fluxo x fluxo adjunto).
        peso = kinf * phi * phi
        peso /= peso.sum()
        self.parcelas = {"Excesso": RHO_EXCESSO}
        for nome, valor in termos.items():
            self.parcelas[nome] = float((peso * valor).sum())
        # O que sobra e das barras, incluindo a deformacao do fluxo que elas causam.
        self.parcelas["Barras"] = self.rho_total - sum(self.parcelas.values())

    # -- resumo ----------------------------------------------------------------
    def quadrantes(self) -> list:
        W = self.pesos_quadrante
        soma = W.sum(axis=1)
        pot_col = self.calor.mean(axis=1)                  # fracao nominal por coluna
        potencia = W @ pot_col / soma
        media = potencia.mean() if potencia.mean() > 1e-12 else 1.0
        h_quente = W @ self.entalpia_saida / soma
        T_quente = estado_da_agua(h_quente)[0]
        T_past = W @ self.T_pastilha.mean(axis=1) / soma
        return [
            {"potencia": float(potencia[q]), "inclinacao": float(potencia[q] / media),
             "T_quente": float(T_quente[q]), "T_fria": float(self.perna_fria[q]),
             "T_pastilha": float(T_past[q])}
            for q in range(4)
        ]

    def qptr(self) -> float:
        if self.potencia_termica < 1e-6:
            return 1.0
        return max(q["inclinacao"] for q in self.quadrantes())

    def resumo(self) -> dict:
        q = self.q_linear
        q_medio = q.mean()
        return {
            "potencia": self.potencia_termica,
            "neutrons": self.n,
            "keff": self.keff,
            "rho": self.rho_total,
            "dolares": self.rho_total * 1e-5 / BETA,
            "sur": self.sur,
            "periodo": 26.06 / self.sur if abs(self.sur) > 1e-3 else float("inf"),
            "parcelas": dict(self.parcelas),
            "boro": self.boro, "boro_alvo": self.boro_alvo,
            "T_entrada": float(self.perna_fria.mean()),
            "T_saida": float(estado_da_agua(self.entalpia_saida.mean())[0]),
            "T_media": float(self.T_agua.mean()),
            "densidade": float(self.densidade.mean()),
            "margem_saturacao": T_SATURACAO - max(float(self.T_agua.max()), float(self.T_saida.max())),
            "T_pastilha": float(self.T_pastilha.mean()),
            "T_centro_max": float(self.T_centro.max()),
            "T_revest_max": float(self.T_revestimento.max()),
            "q_max": float(q.max()),
            "pico": float(q.max() / q_medio) if q_medio > 1e-6 else 0.0,
            "ebulicao_nucleada": bool((self.T_revestimento >= T_SATURACAO - 1e-6).any())
                                 and self.potencia_termica > 1e-3,
            "titulo_max": float(self.titulo.max()),
            "qptr": self.qptr(),
        }

    def temperatura_da_coluna(self, j: int) -> float:
        return float(self.T_pastilha[j].mean())


# ----------------------------------------------------------------------------
# Ligacao com o visualizador (VTK)
# ----------------------------------------------------------------------------
def _altura(limites: tuple, topo: bool) -> float:
    cantos = [
        (limites[i & 1], limites[2 + ((i >> 1) & 1)], limites[4 + ((i >> 2) & 1)])
        for i in range(8)
    ]
    alturas = [sum(c * d for c, d in zip(canto, DIRECAO_VERTICAL)) for canto in cantos]
    return max(alturas) if topo else min(alturas)


def _cruza_no_plano(a: tuple, b: tuple) -> bool:
    return a[0] <= b[1] and b[0] <= a[1] and a[4] <= b[5] and b[4] <= a[5]


# Cores do combustivel pela temperatura media da pastilha
ESCALA_DE_COR = (
    (0.00, (0.62, 0.70, 0.80)),
    (0.35, (0.98, 0.85, 0.30)),
    (0.70, (0.95, 0.45, 0.15)),
    (1.00, (0.80, 0.10, 0.10)),
)
T_COR_MINIMA, T_COR_MAXIMA = T_ENTRADA, 900.0
FONTE_LEITURA = 11
COR_LEITURA = (0.25, 0.25, 0.28)
COR_ALARME = (0.72, 0.18, 0.18)
LARGURA_COLUNA_LEITURA = 230
LARGURA_CELULA, ALTURA_CELULA = 96, 78       # mapa de quadrantes, em pixels


def _cor(temperatura: float) -> tuple:
    t = (temperatura - T_COR_MINIMA) / (T_COR_MAXIMA - T_COR_MINIMA)
    t = min(1.0, max(0.0, t))
    for (a, ca), (b, cb) in zip(ESCALA_DE_COR, ESCALA_DE_COR[1:]):
        if t <= b:
            u = (t - a) / (b - a)
            return tuple(x + u * (y - x) for x, y in zip(ca, cb))
    return ESCALA_DE_COR[-1][1]


def _clarear(cor: tuple, quanto: float = 0.55) -> tuple:
    return tuple(c + (1.0 - c) * quanto for c in cor)


class MonitorDoNucleo:
    """Le as barras do modelo, roda o NucleoPWR e mostra tudo na tela.

    - A cada 100 ms mede, pela caixa de cada barra, quanto dela esta entre o
      topo e o fundo das Wire Things da mesma coluna, e avanca a fisica.
    - Escreve a leitura no lado do painel (acima e abaixo das linhas).
    - Desenha um mapa dos quatro quadrantes no canto do lado do modelo.
    - Pinta cada Wire Thing pela temperatura media da pastilha (esconda o
      vaso no menu Visualizacao ou use o Slider para ver).
    - Dispara o trip das barras se a protecao atuar.
    """

    def __init__(self, partes: list, barras, painel, janela, renderizador_3d=None) -> None:
        import vtk  # so aqui: o NucleoPWR roda sem VTK

        self.vtk = vtk
        self.barras = barras
        self.painel = painel
        self.janela = janela
        self.interator = None
        self.id_temporizador = None
        self.instante_anterior = time.monotonic()
        self.indice_aceleracao = 0
        self.velocidade_real = 1.0
        self.motivo_do_trip = ""
        self.tamanho_desenhado = (0, 0)
        self.quadrante = 0                          # selecionado (0 = Q1)
        self.caidas = set()                         # atores de barras caidas

        # -- regiao dos fuel rods -------------------------------------------
        self.combustivel = [
            p["ator"] for p in partes
            if p["nome"].strip().lower().startswith(PREFIXO_FUEL_RODS)
        ]
        if not self.combustivel:
            raise RuntimeError("Nao achei as Wire Things (fuel rods) no modelo.")
        caixas = [a.GetBounds() for a in self.combustivel]
        self.fundo = min(_altura(c, topo=False) for c in caixas)
        self.topo = max(_altura(c, topo=True) for c in caixas)
        self.altura = max(self.topo - self.fundo, 1e-9)

        centros = [((c[0] + c[1]) / 2.0, (c[4] + c[5]) / 2.0) for c in caixas]
        self.cx = (min(x for x, _ in centros) + max(x for x, _ in centros)) / 2.0
        self.cz = (min(z for _, z in centros) + max(z for _, z in centros)) / 2.0
        distancias = [
            max(abs(a[0] - b[0]), abs(a[1] - b[1]))
            for i, a in enumerate(centros) for b in centros[i + 1:]
        ]
        self.passo = min((d for d in distancias if d > 1e-6), default=1.0)
        posicoes = [((x - self.cx) / self.passo, (z - self.cz) / self.passo) for x, z in centros]

        # -- cada barra do painel cai em uma coluna de combustivel ----------
        self.barras_da_coluna = [[] for _ in self.combustivel]
        self.coluna_da_barra = {}
        for banco in getattr(barras, "bancos", []):
            for ator in banco.atores:
                caixa = ator.GetBounds()
                for j, c in enumerate(caixas):
                    if _cruza_no_plano(caixa, c):
                        self.barras_da_coluna[j].append(ator)
                        self.coluna_da_barra[id(ator)] = j
                        break

        self.posicoes = posicoes
        self.nucleo = NucleoPWR(posicoes, self.medir_insercoes())
        self._preparar_barra_caida()

        # Partida automatica: um botao do painel devolve o controle ao usuario.
        self.automatico = False
        self.situacao = ""              # recado temporario (partida, inicio)
        self.situacao_ate = 0.0
        self.ultimo_tique = time.monotonic()
        acao_do_painel = painel.ao_pressionar

        def pressionar_pelo_usuario(numero, tipo):
            if self.automatico:
                self._encerrar_partida("interrompida pelo operador")
            if acao_do_painel is not None:
                acao_do_painel(numero, tipo)

        painel.ao_pressionar = pressionar_pelo_usuario

        # A cor passa a ser a da temperatura: tira a textura e as cores por
        # vertice que vieram do .glb e o brilho metalico que escurece tudo.
        for ator in self.combustivel:
            propriedade = ator.GetProperty()
            propriedade.RemoveAllTextures()
            propriedade.SetMetallic(0.0)
            propriedade.SetRoughness(0.6)
            if ator.GetMapper() is not None:
                ator.GetMapper().ScalarVisibilityOff()

        self._criar_leitura()
        self._criar_mapa(renderizador_3d)
        self._criar_rotulos_3d(renderizador_3d)

        janela.AddObserver("StartEvent", self._antes_de_renderizar)
        self._dispor()
        self.iniciar_reator(MODO_INICIAL, renderizar=False)

    # -- inicio da simulacao ---------------------------------------------------
    def _passos_totais(self, banco) -> int:
        import sys
        modulo = sys.modules.get(type(banco).__module__)
        return int(getattr(modulo, "PASSOS_TOTAIS", 231))

    def _posicionar_bancos(self, posicoes: dict) -> None:
        for banco in self.barras.bancos:
            total = self._passos_totais(banco)
            banco.sentido = 0
            banco.acumulado = 0.0
            banco.travado = False
            banco.passos = float(min(total, max(0, posicoes.get(banco.sigla, 0))))
            banco.aplicar()

    def iniciar_reator(self, modo: str = MODO_INICIAL, renderizar: bool = True) -> None:
        """Comeca (ou recomeca) a simulacao: "operando", "partida" ou "desligado"."""
        barras = self.barras
        barras.em_trip = False
        self.caidas.clear()
        self.motivo_do_trip = ""
        self.automatico = False
        self.situacao = ""

        if modo == "operando":
            self._posicionar_bancos(POSICOES_DE_OPERACAO)
        elif modo == "partida":
            self._posicionar_bancos({b.sigla: (self._passos_totais(b) if b.tipo == "desligamento" else 0)
                                     for b in barras.bancos})
        else:
            self._posicionar_bancos({})           # tudo no fundo
        barras.atualizar_indicadores(renderizar=False)

        self.nucleo = NucleoPWR(self.posicoes, self.medir_insercoes())
        if modo == "operando":
            if not self.nucleo.equilibrar(POTENCIA_INICIAL, xenonio=XENONIO_DE_EQUILIBRIO):
                self._avisar("sem boro crítico: ajuste as barras", 30.0)
        elif modo == "partida":
            self.automatico = True
            self.situacao = "partida automática"

        self.instante_anterior = time.monotonic()
        self._atualizar_tela(renderizar=renderizar)

    def _encerrar_partida(self, motivo: str) -> None:
        for banco in self.barras.bancos:
            if banco.sentido > 0:
                self.barras.soltar(banco.linha, "out")
        self.automatico = False
        self._avisar(f"partida {motivo}")

    def _avisar(self, texto: str, segundos: float = 15.0) -> None:
        self.situacao = texto
        self.situacao_ate = time.monotonic() + segundos

    def estado_do_reator(self, r: dict) -> str:
        """Frase curta do estado atual, recalculada a cada atualizacao."""
        if self.automatico:
            return self.situacao.upper()
        if self.barras.em_trip:
            return "DESLIGADO POR TRIP"
        potencia, sur = r["potencia"], r["sur"]
        if all(b.inserida for b in self.barras.bancos) and r["rho"] < -1000.0:
            estado = "DESLIGADO: barras no fundo"
        elif potencia >= 0.01:
            tendencia = "subindo" if sur > 0.05 else ("caindo" if sur < -0.05 else "estável")
            estado = f"EM POTÊNCIA {potencia * 100:.1f} %, {tendencia}"
        elif r["keff"] < 0.998:
            estado = f"SUBCRÍTICO (keff {r['keff']:.4f})"
        elif sur > 0.05:
            estado = "SUPERCRÍTICO: potência subindo"
        else:
            estado = "CRÍTICO em potência zero"
        if self.situacao and time.monotonic() < self.situacao_ate:
            estado += f"  ({self.situacao})"
        return estado

    def _partida_automatica(self) -> None:
        """Retira os bancos em ordem, esperando quando a SUR passa do limite."""
        if not self.automatico:
            return
        barras, nucleo = self.barras, self.nucleo
        if barras.em_trip:
            self._encerrar_partida("interrompida pelo trip")
            return
        alvo = POTENCIA_ALVO_DA_PARTIDA
        fila = [b for b in sorted(barras.bancos, key=lambda b: b.linha) if not b.retirada]
        if nucleo.potencia_termica >= 0.98 * alvo or not fila:
            self._encerrar_partida("concluída")
            return
        banco = fila[0]
        for outro in barras.bancos:
            if outro is not banco and outro.sentido > 0:
                barras.soltar(outro.linha, "out")
        # Perto do alvo so anda com a potencia quase parada.
        limite = SUR_MAXIMO_DA_PARTIDA if nucleo.potencia_termica < 0.9 * alvo else 0.2
        if nucleo.sur < limite and banco.sentido == 0:
            barras.pressionar(banco.linha, "out")
        elif nucleo.sur >= limite and banco.sentido > 0:
            barras.soltar(banco.linha, "out")
        self.situacao = f"partida automática: {banco.sigla}"

    # -- montagem da tela ----------------------------------------------------
    def _texto(self, renderizador, tamanho=FONTE_LEITURA, cor=COR_LEITURA, negrito=False):
        texto = self.vtk.vtkTextActor()
        prop = texto.GetTextProperty()
        prop.SetFontFamilyToCourier()
        prop.SetFontSize(tamanho)
        prop.SetColor(*cor)
        prop.SetBold(negrito)
        prop.SetLineSpacing(1.15)
        texto.SetLayerNumber(2)
        renderizador.AddViewProp(texto)
        return texto

    def _criar_leitura(self) -> None:
        renderizador = self.painel.renderizador
        self.textos = [self._texto(renderizador) for _ in range(4)]
        for texto in self.textos[:2]:
            texto.GetTextProperty().SetVerticalJustificationToTop()
        self.alarme = self._texto(renderizador, cor=COR_ALARME, negrito=True)
        self.estado = self._texto(renderizador, tamanho=FONTE_LEITURA + 1,
                                  cor=(0.12, 0.28, 0.50), negrito=True)

    def _criar_mapa(self, renderizador_3d) -> None:
        from dropdown import Retangulo2D

        vtk = self.vtk
        self.renderizador_mapa = vtk.vtkRenderer()
        self.renderizador_mapa.SetLayer(1)
        self.renderizador_mapa.InteractiveOff()
        if renderizador_3d is not None:
            self.renderizador_mapa.SetViewport(*renderizador_3d.GetViewport())
        else:
            x0, _, x1, _ = self.painel.renderizador.GetViewport()
            self.renderizador_mapa.SetViewport(*((0.5, 0.0, 1.0, 1.0) if x0 < 0.25 else (0.0, 0.0, 0.5, 1.0)))
        self.janela.AddRenderer(self.renderizador_mapa)

        self.celulas, self.textos_celula = [], []
        for _ in range(4):
            celula = Retangulo2D()
            celula.definir_camada(0)
            celula.adicionar_em(self.renderizador_mapa)
            texto = self._texto(self.renderizador_mapa)
            texto.GetTextProperty().SetVerticalJustificationToTop()
            self.celulas.append(celula)
            self.textos_celula.append(texto)
        self.titulo_mapa = self._texto(self.renderizador_mapa)

    def _criar_rotulos_3d(self, renderizador_3d) -> None:
        self.rotulos_3d = []
        if renderizador_3d is None or not hasattr(self.vtk, "vtkBillboardTextActor3D"):
            return
        altura = self.topo + 0.25 * self.altura
        for q, (sx, sz) in enumerate(SINAIS_DOS_QUADRANTES):
            rotulo = self.vtk.vtkBillboardTextActor3D()
            rotulo.SetInput(f"Q{q + 1}")
            prop = rotulo.GetTextProperty()
            prop.SetFontSize(18)
            prop.SetBold(True)
            prop.SetColor(0.15, 0.30, 0.55)
            prop.SetJustificationToCentered()
            ponto = [0.0, 0.0, 0.0]
            ponto[0] = self.cx + sx * 1.4 * self.passo
            ponto[2] = self.cz + sz * 1.4 * self.passo
            ponto = [p + altura * d for p, d in zip(ponto, DIRECAO_VERTICAL)]
            rotulo.SetPosition(*ponto)
            renderizador_3d.AddActor(rotulo)
            self.rotulos_3d.append(rotulo)

    # -- barra caida ----------------------------------------------------------
    def _preparar_barra_caida(self) -> None:
        """Escolhe uma barra por quadrante e ensina o banco a deixa-la no fundo."""
        W = self.nucleo.pesos_quadrante
        self.barra_do_quadrante = [None] * 4
        bancos = list(getattr(self.barras, "bancos", []))
        preferidos = [b for b in bancos if b.sigla == BANCO_DA_BARRA_CAIDA] + bancos
        for q in range(4):
            for banco in preferidos:
                for ator in banco.atores:
                    j = self.coluna_da_barra.get(id(ator))
                    if j is not None and W[q, j] >= 0.999:
                        self.barra_do_quadrante[q] = (banco, ator)
                        break
                if self.barra_do_quadrante[q]:
                    break

        for banco in bancos:
            original = banco.aplicar

            def aplicar(banco=banco, original=original):
                original()
                for i, ator in enumerate(banco.atores):
                    if id(ator) in self.caidas:
                        deslocamento = [-c * banco.curso for c in DIRECAO_VERTICAL] + [0.0]
                        x, y, z, _ = banco.para_o_ator[i].MultiplyPoint(deslocamento)
                        ator.SetPosition(x, y, z)

            banco.aplicar = aplicar

    def alternar_barra_caida(self) -> None:
        escolha = self.barra_do_quadrante[self.quadrante]
        if escolha is None:
            return
        banco, ator = escolha
        if id(ator) in self.caidas:
            self.caidas.discard(id(ator))
        else:
            self.caidas.add(id(ator))
        banco.aplicar()

    # -- barras -> fracao inserida no combustivel ---------------------------
    def medir_insercoes(self) -> list:
        insercoes = []
        for atores in self.barras_da_coluna:
            fracao = 0.0
            for ator in atores:
                fundo_da_barra = _altura(ator.GetBounds(), topo=False)
                dentro = (self.topo - fundo_da_barra) / self.altura
                fracao = max(fracao, min(1.0, max(0.0, dentro)))
            insercoes.append(fracao)
        return insercoes

    # -- relogio e teclado ---------------------------------------------------
    def conectar(self, interator) -> None:
        self.interator = interator
        interator.AddObserver("TimerEvent", self._ao_temporizador)
        interator.AddObserver("CharEvent", self._caractere)
        # No Windows um timer criado antes de a janela existir nunca dispara
        # (o VTK pede o timer ao sistema sem janela). Por isso inicializa o
        # interator antes; chamar Initialize de novo depois nao faz nada.
        if not interator.GetInitialized():
            interator.Initialize()
        self._criar_temporizador()

    def _criar_temporizador(self) -> None:
        if self.interator is None:
            return
        if self.id_temporizador is not None:
            self.interator.DestroyTimer(self.id_temporizador)
        self.instante_anterior = time.monotonic()
        self.ultimo_tique = time.monotonic()
        self.id_temporizador = self.interator.CreateRepeatingTimer(INTERVALO_DA_FISICA)

    def _conferir_temporizador(self) -> None:
        """Se o relogio da fisica parou de chegar, cria outro."""
        if self.interator is None or self.id_temporizador is None:
            return
        if time.monotonic() - self.ultimo_tique > 1.0:
            self._criar_temporizador()

    def _caractere(self, obj, evento) -> None:
        self._conferir_temporizador()
        codigo = obj.GetKeyCode() or ""
        nome = (obj.GetKeySym() or "").lower()
        if nome.startswith("kp_") and nome[3:].isdigit():
            codigo = nome[3:]
        if codigo in ("1", "2", "3", "4"):
            self.quadrante = int(codigo) - 1
        elif codigo in ("j", "J"):
            self.nucleo.mudar_boro(-PASSO_DO_BORO)
        elif codigo in ("k", "K"):
            self.nucleo.mudar_boro(+PASSO_DO_BORO)
        elif codigo in ("x", "X"):
            self.indice_aceleracao = (self.indice_aceleracao + 1) % len(ACELERACOES)
        elif codigo in ("d", "D"):
            self.alternar_barra_caida()
        elif codigo in ("i", "I"):
            self.iniciar_reator(MODO_INICIAL)
            return
        elif codigo == "," or nome == "comma":
            self.nucleo.mudar_perna_fria(self.quadrante, -PASSO_DA_PERNA_FRIA)
        elif codigo == "." or nome == "period":
            self.nucleo.mudar_perna_fria(self.quadrante, +PASSO_DA_PERNA_FRIA)
        else:
            return
        self._atualizar_tela()

    def _ao_temporizador(self, obj, evento) -> None:
        # Todo timer da janela (barras, menu, este) dispara o mesmo TimerEvent,
        # e o VTK nem sempre informa qual foi. Por isso a fisica anda pelo
        # relogio: roda quando ja passou um intervalo, venha o evento de onde vier.
        if self.id_temporizador is None:
            return
        agora = time.monotonic()
        decorrido = agora - self.instante_anterior
        if decorrido < 0.9 * INTERVALO_DA_FISICA / 1000.0:
            return
        real = min(1.0, decorrido)     # se a tela travar um pouco, recupera
        self.instante_anterior = agora
        self.ultimo_tique = agora

        aceleracao = ACELERACOES[self.indice_aceleracao]
        self.nucleo.passo_termico = min(1.0, 0.05 * aceleracao)
        self._partida_automatica()
        self.nucleo.definir_insercoes(self.medir_insercoes())
        simulado = self.nucleo.avancar(real * aceleracao, limite_de_cpu=ORCAMENTO_POR_QUADRO)
        self.velocidade_real += (simulado / max(decorrido, 1e-3) - self.velocidade_real) * 0.2
        self._protecao()
        self._atualizar_tela()

    def _protecao(self) -> None:
        if not self.barras.em_trip:
            self.motivo_do_trip = ""
        if not TRIP_AUTOMATICO or self.barras.em_trip:
            return
        motivo = ""
        if self.nucleo.pico_de_neutrons >= LIMITE_DE_POTENCIA:
            motivo = f"potência alta ({LIMITE_DE_POTENCIA * 100:.0f} %)"
        elif TRIP_NA_SATURACAO and float(self.nucleo.T_saida.max()) >= T_SATURACAO - 0.05:
            motivo = "água do canal na saturação"
        if motivo:
            self.motivo_do_trip = motivo
            self.barras.trip()

    # -- desenho ----------------------------------------------------------
    def _antes_de_renderizar(self, obj, evento) -> None:
        self._conferir_temporizador()
        if self.janela.GetSize() != self.tamanho_desenhado:
            self._dispor()

    def _dispor(self) -> None:
        from dropdown import ALTURA_BARRA

        largura, altura = self.janela.GetSize()
        self.tamanho_desenhado = (largura, altura)
        x0, _, x1, _ = self.painel.renderizador.GetViewport()
        esquerda, direita = x0 * largura, x1 * largura
        total = 2 * LARGURA_COLUNA_LEITURA
        inicio = esquerda + max(12.0, (direita - esquerda - total) / 2.0)
        topo = altura - ALTURA_BARRA - 10
        self.textos[0].SetDisplayPosition(int(inicio), int(topo))
        self.textos[1].SetDisplayPosition(int(inicio + LARGURA_COLUNA_LEITURA), int(topo))
        self.textos[2].SetDisplayPosition(int(inicio), 26)
        self.textos[3].SetDisplayPosition(int(inicio + LARGURA_COLUNA_LEITURA), 26)
        self.alarme.SetDisplayPosition(int(inicio), 6)
        # Estado do reator: logo abaixo do bloco de cima (6 linhas).
        self.estado.SetDisplayPosition(int(inicio), int(topo - 6 * 15 - 16))

        # Mapa: canto de baixo, do lado do modelo, perto do painel.
        mx0, _, mx1, _ = self.renderizador_mapa.GetViewport()
        if mx0 >= x1 - 1e-6:
            base_x = mx0 * largura + 12
        else:
            base_x = mx1 * largura - 12 - 2 * LARGURA_CELULA
        base_y = 10
        # Q2 Q1 em cima, Q3 Q4 embaixo (vista de cima).
        cantos = {0: (1, 1), 1: (0, 1), 2: (0, 0), 3: (1, 0)}
        for q, (cx, cy) in cantos.items():
            x = base_x + cx * LARGURA_CELULA
            y = base_y + cy * ALTURA_CELULA
            self.celulas[q].definir_area(x, y, x + LARGURA_CELULA - 3, y + ALTURA_CELULA - 3)
            self.textos_celula[q].SetDisplayPosition(int(x + 5), int(y + ALTURA_CELULA - 7))
        self.titulo_mapa.SetDisplayPosition(int(base_x), int(base_y + 2 * ALTURA_CELULA + 2))

    def _atualizar_tela(self, renderizar: bool = True) -> None:
        r = self.nucleo.resumo()
        aceleracao = ACELERACOES[self.indice_aceleracao]

        if r["neutrons"] >= 1e-3:
            neutrons = f"{r['neutrons'] * 100:9.2f} %"
        else:
            neutrons = f"{r['neutrons']:9.2e}"
        periodo = "  inf" if math.isinf(r["periodo"]) or abs(r["periodo"]) > 999 else f"{r['periodo']:5.0f}"
        velocidade = f"x{aceleracao}"
        if aceleracao > 1:
            velocidade += f" (real x{self.velocidade_real:.0f})"

        self.textos[0].SetInput(
            f"NÚCLEO  {velocidade} X\n"
            f"Potência {r['potencia'] * 100:6.2f} % {r['potencia'] * POTENCIA_NOMINAL / 1e6:5.0f} MW\n"
            f"Nêutrons {neutrons}\n"
            f"SUR {r['sur']:+6.2f} dpm  T{periodo} s\n"
            f"keff     {r['keff']:.5f}\n"
            f"Reat. {r['rho']:+6.0f} pcm {r['dolares']:+.2f}$"
        )
        partes = r["parcelas"]
        boro = f"{r['boro']:.0f}"
        if abs(r["boro_alvo"] - r["boro"]) >= 0.5:
            boro += f">{r['boro_alvo']:.0f}"
        self.textos[1].SetInput(
            f"REAT. pcm  (excesso {RHO_EXCESSO:+.0f})\n"
            f"Barras    {partes['Barras']:+7.0f}\n"
            f"Boro      {partes['Boro']:+7.0f} {boro} ppm\n"
            f"Moderador {partes['Moderador']:+7.0f}\n"
            f"Doppler   {partes['Doppler']:+7.0f}\n"
            f"Xenônio   {partes['Xenônio']:+7.0f}"
        )
        self.textos[2].SetInput(
            f"ÁGUA {PRESSAO_MPA:.1f} MPa (J/K boro)\n"
            f"Entrada  {r['T_entrada']:6.1f} °C\n"
            f"Saída    {r['T_saida']:6.1f} °C\n"
            f"Média    {r['T_media']:6.1f} °C\n"
            f"Densid.  {r['densidade']:6.1f} kg/m³\n"
            f"Margem sat. {r['margem_saturacao']:5.1f} °C"
        )
        self.textos[3].SetInput(
            f"COMBUSTÍVEL\n"
            f"Pastilha méd. {r['T_pastilha']:6.0f} °C\n"
            f"Centro máx.   {r['T_centro_max']:6.0f} °C\n"
            f"Revest. máx.  {r['T_revest_max']:6.0f} °C\n"
            f"q' máx.     {r['q_max'] / 1e3:6.1f} kW/m\n"
            f"Pico Fq       {r['pico']:6.2f}"
        )

        # -- quadrantes --------------------------------------------------------
        quadrantes = self.nucleo.quadrantes()
        self.titulo_mapa.SetInput(
            f"QUADRANTES  QPTR {r['qptr']:.3f}\n1-4 escolhe  D barra  , . laço")
        caida_em = set()
        for q, escolha in enumerate(self.barra_do_quadrante):
            if escolha and id(escolha[1]) in self.caidas:
                caida_em.add(q)
        for q, dados in enumerate(quadrantes):
            cor = _clarear(_cor(dados["T_pastilha"]))
            selecionado = q == self.quadrante
            self.celulas[q].pintar(cor, (0.15, 0.15, 0.2) if selecionado else (0.6, 0.6, 0.62))
            self.celulas[q].ator_contorno.GetProperty().SetLineWidth(3.0 if selecionado else 1.0)
            marca = " BARRA" if q in caida_em else ""
            desvio = dados["T_fria"] - T_ENTRADA
            self.textos_celula[q].SetInput(
                f"Q{q + 1}{marca}\n"
                f"{dados['potencia'] * 100:6.1f} %\n"
                f"Tq {dados['T_quente']:6.1f}\n"
                f"Tf {dados['T_fria']:6.1f}" + (f"{desvio:+.0f}" if abs(desvio) >= 0.5 else "") + "\n"
                f"Pa {dados['T_pastilha']:6.0f}"
            )

        self.estado.SetInput(self.estado_do_reator(r))

        alarmes = []
        if self.barras.em_trip:
            alarmes.append("TRIP" + (f": {self.motivo_do_trip}" if self.motivo_do_trip else ""))
        if r["T_centro_max"] >= T_FUSAO_UO2:
            alarmes.append("FUSÃO DO UO2")
        if r["titulo_max"] > 0.0:
            alarmes.append("EBULIÇÃO NO CANAL")
        elif r["ebulicao_nucleada"]:
            alarmes.append("ebulição nucleada")
        if 0.0 < r["periodo"] < 10.0:
            alarmes.append("PERÍODO CURTO")
        if r["qptr"] > LIMITE_QPTR and r["potencia"] > 0.05:
            alarmes.append(f"QPTR {r['qptr']:.3f}")
        if caida_em:
            alarmes.append("BARRA CAÍDA " + " ".join(f"Q{q + 1}" for q in sorted(caida_em)))
        self.alarme.SetInput("  ".join(alarmes))

        for j, ator in enumerate(self.combustivel):
            ator.GetProperty().SetColor(*_cor(self.nucleo.temperatura_da_coluna(j)))

        if renderizar:
            self.janela.Render()


# ----------------------------------------------------------------------------
# Teste sem VTK
# ----------------------------------------------------------------------------
def _grade_do_modelo() -> list:
    """As 21 posicoes do modelo: 5x5 sem os quatro cantos."""
    return [(i, j) for i in range(-2, 3) for j in range(-2, 3) if abs(i) + abs(j) < 4]


def _teste() -> None:
    posicoes = _grade_do_modelo()
    aneis = {"SA": [(1, 2)], "SB": [(0, 2)], "CA": [(1, 1)], "CB": [(0, 0), (0, 1)]}

    def banco(p):
        anel = tuple(sorted((abs(p[0]), abs(p[1]))))
        return next(s for s, a in aneis.items() if anel in a)

    bancos = [banco(p) for p in posicoes]
    nucleo = NucleoPWR(posicoes, [1.0] * len(posicoes))

    print("Valor de cada banco sozinho, a partir de tudo fora (pcm):")
    for sigla in aneis:
        fora = NucleoPWR(posicoes, [0.0] * len(posicoes)).rho_total
        so_este = NucleoPWR(posicoes, [1.0 if b == sigla else 0.0 for b in bancos]).rho_total
        print(f"   {sigla}: {so_este - fora:8.0f}")

    posicao = {s: 1.0 for s in aneis}
    velocidade = {"SA": 64, "SB": 64, "CA": 48, "CB": 48}   # passos por minuto

    def linha(rotulo):
        r = nucleo.resumo()
        q = nucleo.quadrantes()
        print(f"{nucleo.tempo:7.0f} s  {rotulo:<24} P={r['potencia'] * 100:7.2f} %  "
              f"keff={r['keff']:.5f}  Tsai={r['T_saida']:6.1f}  Tc={r['T_centro_max']:5.0f}  "
              f"QPTR={r['qptr']:.3f}  Q=" + " ".join(f"{d['potencia'] * 100:5.1f}" for d in q))

    def insercoes():
        return [posicao[b] for b in bancos]

    linha("barras no fundo")
    for sigla in ("SA", "SB", "CA", "CB"):
        while posicao[sigla] > 0.0:
            posicao[sigla] = max(0.0, posicao[sigla] - velocidade[sigla] / 60.0 / 231.0)
            nucleo.definir_insercoes(insercoes())
            nucleo.avancar(1.0)
        linha(f"{sigla} retirado")
    nucleo.avancar(600.0)
    linha("+10 min")

    print("\nBarra do CA caida no quadrante Q1:")
    q1 = next(j for j, p in enumerate(posicoes) if banco(p) == "CA" and p[0] > 0 and p[1] < 0)
    caida = insercoes()
    caida[q1] = 1.0
    nucleo.definir_insercoes(caida)
    anterior = 0
    for t in (1, 10, 60, 300):
        nucleo.avancar(t - anterior)
        anterior = t
        linha(f"caida +{t} s")
    nucleo.definir_insercoes(insercoes())
    nucleo.avancar(600.0)
    linha("realinhada +10 min")

    print("\nPerna fria do laco 3 +5 C (gerador de vapor 3 tirando menos calor):")
    nucleo.mudar_perna_fria(2, 5.0)
    anterior = 0
    for t in (10, 60, 300):
        nucleo.avancar(t - anterior)
        anterior = t
        linha(f"laco 3 quente +{t} s")
    q = nucleo.quadrantes()
    for i, d in enumerate(q):
        print(f"   Q{i + 1}: fria {d['T_fria']:.1f}  quente {d['T_quente']:.1f}  "
              f"pastilha {d['T_pastilha']:.0f}  inclinacao {d['inclinacao']:.3f}")
    nucleo.mudar_perna_fria(2, -5.0)

    print("\nTrip:")
    nucleo.definir_insercoes([1.0] * len(posicoes))
    anterior = 0
    for t in (1, 10, 100):
        nucleo.avancar(t - anterior)
        anterior = t
        linha(f"trip +{t} s")

    t0 = time.perf_counter()
    nucleo.avancar(5.0)
    print(f"\nCusto: {(time.perf_counter() - t0) / 100 * 1e3:.1f} ms por passo de 50 ms")


if __name__ == "__main__":
    _teste()