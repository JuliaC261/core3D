#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core3D - visualizador 3D do nucleo de um PWR (modelo .glb, glTF binario).

A janela se chama "Core3D" e usa o icone core3d.ico (Windows, inclusive na
barra de tarefas) ou core3d.png (Linux), que precisam estar na mesma pasta
deste arquivo (sem eles o programa abre normalmente, so sem icone).

O modelo aparece no lado direito da janela, sobre fundo branco, e pode ser
girado livremente com o botao esquerdo do mouse, com zoom na roda.

No topo da tela ha o menu "Visualizacao": clicando nele abre uma lista com as
partes do modelo. Cada parte acesa (azul) esta visivel; clicando nela a cor
some e a peca fica invisivel. Clicando de novo ela volta.

O menu fica todo no arquivo menu_visualizacao.py, que precisa estar na mesma
pasta deste aqui.

Requisito: pip install vtk

Uso:
    python visualizador_3d.py
    python visualizador_3d.py caminho/para/outro_modelo.glb
"""

from __future__ import annotations

import json
import os
import struct
import sys
import time

import vtk

from dropdown import MenuVisualizacao
from painel import PainelDeControle
from nucleo import MonitorDoNucleo

# ----------------------------------------------------------------------------
# Configuracoes gerais
# ----------------------------------------------------------------------------
ARQUIVO_PADRAO = "Yotuber_Nuclear_Reacts_To_Demon_Core.glb"
NOME_DO_PROGRAMA = "Core3D"
ARQUIVO_DO_ICONE = "core3d.png"      # na mesma pasta do script

LARGURA, ALTURA = 1100, 680          # tamanho inicial da janela
COR_FUNDO = (1.0, 1.0, 1.0)          # branco
LADO_DO_MODELO = "direita"           # "direita" ou "esquerda"
SENSIBILIDADE = 10.0                 # velocidade do giro com o mouse
ZOOM_RODA = 0.5                      # sensibilidade da roda (maior = mais rapido)
ZOOM_TECLADO = 1.1                   # quanto cada toque em + / - aproxima
PASSO_TECLADO = 5.0                  # graus por toque nas setas
ANGULO_INICIAL = 25.0                # giro inicial da camera, em graus
ELEVACAO_CAMERA = 10.0               # olhar um pouco de cima (graus)
ZOOM = 1.1                           # >1 aproxima, <1 afasta
MOSTRAR_INSTRUCOES = False           # texto de ajuda (o painel ocupa esse espaco)

# ----------------------------------------------------------------------------
# Barras de controle, no esquema KWU (Siemens) de Angra 2
#
# Angra 2 tem 61 barras "pretas" (Ag-In-Cd), divididas em dois tipos de banco:
#   L  - banco L: quase todas as barras. Em operacao fica fora (no maximo um
#        pouco dentro, para acertar a distribuicao axial de potencia); e o que
#        mantem o reator desligado. Todas as barras dele andam juntas.
#   D  - grupos D, de 4 barras cada, que controlam a potencia (compensam o
#        Doppler). Andam em serie, sem sobreposicao: um grupo so sai depois
#        que o anterior saiu todo. Em potencia so o ultimo fica um pouco dentro.
# A usina tem 4 grupos D; o painel tem 4 linhas, entao aqui ficam L + D1..D3
# (as barras do quarto grupo D contam no banco L, ver nucleo.py).
#
# O modelo tem uma barra por coluna (21). Cada banco pega as barras de certos
# "aneis" da grade. O anel e o par (menor, maior) da distancia ate o centro,
# em passos da grade:
#     (0,0) centro   (0,1) cruz interna   (1,1) diagonais
#     (0,2) cruz externa   (1,2) as oito posicoes restantes da borda
# Assim todo banco tem simetria de 90 graus (e de 1/8 do nucleo), e cada grupo
# D tem 4 barras, como na usina.
#
# O painel funciona como a chave seletora na posicao de banco individual:
# cada linha move um banco com o seu IN-HOLD-OUT. Segurou, anda; soltou, para.
# ----------------------------------------------------------------------------
BANCOS = [
    # linha, sigla, tipo,          aneis da grade
    (1, "L",  "desligamento", [(1, 2), (0, 0)]),  # 9 barras (borda + centro)
    (2, "D1", "controle",     [(0, 2)]),          # 4 barras (cruz externa)
    (3, "D2", "controle",     [(1, 1)]),          # 4 barras (diagonais)
    (4, "D3", "controle",     [(0, 1)]),          # 4 barras (cruz interna); regula
]
PREFIXO_DAS_BARRAS = "support"       # barras = Supports que passam da Top Platform
NOME_TOPO_DO_REATOR = "top platform"

# Posicao contada em passos, como nos contadores de passos da sala de controle:
# 0 = toda inserida, PASSOS_TOTAIS = toda retirada.
PASSOS_TOTAIS = 231
PASSOS_POR_MINUTO = {"desligamento": 64, "controle": 48}
ACELERACAO_DO_TEMPO = 1.0            # 1 = tempo real; 5 = cinco vezes mais rapido
COMECAR_INSERIDAS = True             # reator desligado: tudo no fundo

# Procedimento (KWU): os grupos D so saem com o banco L quase todo fora (o L
# pode ficar um pouco dentro, como no controle axial da usina), e um grupo D
# so sai depois que o anterior saiu todo (em serie, sem sobreposicao).
# Inserir e sempre permitido.
EXIGIR_SEQUENCIA = True
FRACAO_MINIMA_DO_L = 0.90            # fracao retirada do L para liberar os D

# Trip (tecla T): as garras soltam e as barras caem por gravidade. Elas entram
# no amortecedor (dashpot) perto do fundo e terminam a descida devagar.
TEMPO_ATE_O_DASHPOT = 2.4            # segundos, partindo de toda retirada
FRACAO_DO_DASHPOT = 0.10             # trecho final do curso freado pelo dashpot
TEMPO_NO_DASHPOT = 0.6               # segundos para vencer esse trecho final

# Ate onde a barra entra:
#   "fuel rods"      -> fundo da barra rente ao fundo das Wire Things (fuel rods)
#   "base interna"   -> fundo da barra encostado no topo da Internal Platform Bottom
FIM_DA_INSERCAO = "fuel rods"
PREFIXO_FUEL_RODS = "wire things"
NOME_BASE_INTERNA = "internal platform bottom"
DIRECAO_DA_PECA = (0.0, 1.0, 0.0)    # "para cima" no mundo (sentido de retirar)
INTERVALO_DA_BARRA = 16              # milissegundos entre quadros do movimento

# ----------------------------------------------------------------------------
# Localizar o arquivo do modelo
# ----------------------------------------------------------------------------
def localizar_modelo(argumentos: list[str]) -> str:
    """Devolve o caminho do .glb informado na linha de comando ou o padrao."""
    if argumentos:
        caminho = argumentos[0]
    else:
        pasta_do_script = os.path.dirname(os.path.abspath(__file__))
        caminho = os.path.join(pasta_do_script, ARQUIVO_PADRAO)

    if not os.path.isfile(caminho):
        sys.exit(
            f"Arquivo nao encontrado: {caminho}\n"
            "Coloque o .glb na mesma pasta do script ou passe o caminho:\n"
            "    python visualizador_3d.py caminho/do/modelo.glb"
        )
    return caminho


# ----------------------------------------------------------------------------
# Nomes das partes
#
# O VTK importa a geometria mas descarta os nomes dos objetos. Entao eu leio o
# JSON de dentro do .glb, calculo a caixa que envolve cada peca e caso com as
# caixas dos atores importados. Se algo nao bater, vira "Parte N".
# ----------------------------------------------------------------------------
def _ler_json_do_glb(caminho: str) -> dict | None:
    """Extrai o bloco JSON de um arquivo .glb."""
    try:
        with open(caminho, "rb") as arquivo:
            if arquivo.read(4) != b"glTF":
                return None
            arquivo.read(8)  # versao e tamanho total
            tamanho, tipo = struct.unpack("<II", arquivo.read(8))
            if tipo != 0x4E4F534A:  # "JSON"
                return None
            return json.loads(arquivo.read(tamanho))
    except (OSError, ValueError, struct.error):
        return None


def _multiplicar(a: list, b: list) -> list:
    """Multiplica duas matrizes 4x4 guardadas como listas de listas."""
    return [
        [sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)]
        for i in range(4)
    ]


def _matriz_do_no(no: dict) -> list:
    """Monta a matriz 4x4 de um no do glTF (matrix ou translacao/rotacao/escala)."""
    if "matrix" in no:
        m = no["matrix"]  # vem em ordem de coluna
        return [[m[0], m[4], m[8], m[12]],
                [m[1], m[5], m[9], m[13]],
                [m[2], m[6], m[10], m[14]],
                [m[3], m[7], m[11], m[15]]]

    tx, ty, tz = no.get("translation", [0.0, 0.0, 0.0])
    x, y, z, w = no.get("rotation", [0.0, 0.0, 0.0, 1.0])
    sx, sy, sz = no.get("scale", [1.0, 1.0, 1.0])

    rotacao = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    escala = (sx, sy, sz)
    return [
        [rotacao[i][0] * escala[0],
         rotacao[i][1] * escala[1],
         rotacao[i][2] * escala[2],
         (tx, ty, tz)[i]]
        for i in range(3)
    ] + [[0.0, 0.0, 0.0, 1.0]]


def _caixa_da_malha(gltf: dict, indice_malha: int) -> tuple | None:
    """Caixa (minimos e maximos) de uma malha, em coordenadas locais."""
    menor = [float("inf")] * 3
    maior = [float("-inf")] * 3
    for primitiva in gltf["meshes"][indice_malha].get("primitives", []):
        indice = primitiva.get("attributes", {}).get("POSITION")
        if indice is None:
            continue
        acessor = gltf["accessors"][indice]
        if "min" not in acessor or "max" not in acessor:
            continue
        for eixo in range(3):
            menor[eixo] = min(menor[eixo], acessor["min"][eixo])
            maior[eixo] = max(maior[eixo], acessor["max"][eixo])
    if any(v == float("inf") for v in menor):
        return None
    return tuple(menor), tuple(maior)


def _caixas_dos_nos(gltf: dict) -> list:
    """Percorre a cena e devolve [(nome, caixa em coordenadas do mundo), ...]."""
    identidade = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    cena = gltf.get("scenes", [{}])[gltf.get("scene", 0)]
    encontrados = []

    def visitar(indice_no: int, matriz_pai: list) -> None:
        no = gltf["nodes"][indice_no]
        matriz = _multiplicar(matriz_pai, _matriz_do_no(no))

        if "mesh" in no:
            caixa = _caixa_da_malha(gltf, no["mesh"])
            if caixa is not None:
                menor, maior = caixa
                limites = [float("inf"), float("-inf")] * 3
                for i in range(8):
                    canto = (
                        maior[0] if i & 1 else menor[0],
                        maior[1] if i & 2 else menor[1],
                        maior[2] if i & 4 else menor[2],
                    )
                    for eixo in range(3):
                        valor = sum(matriz[eixo][k] * canto[k] for k in range(3))
                        valor += matriz[eixo][3]
                        limites[eixo * 2] = min(limites[eixo * 2], valor)
                        limites[eixo * 2 + 1] = max(limites[eixo * 2 + 1], valor)
                nome = no.get("name") or gltf["meshes"][no["mesh"]].get("name")
                encontrados.append((nome, tuple(limites)))

        for filho in no.get("children", []):
            visitar(filho, matriz)

    for raiz in cena.get("nodes", range(len(gltf.get("nodes", [])))):
        visitar(raiz, identidade)
    return encontrados


def nomes_das_partes(caminho: str, atores: list) -> list[str]:
    """Casa cada ator importado com o nome da peca correspondente no .glb."""
    nomes = [f"Parte {i + 1}" for i in range(len(atores))]

    gltf = _ler_json_do_glb(caminho)
    if not gltf or "nodes" not in gltf:
        return nomes

    try:
        candidatos = _caixas_dos_nos(gltf)
    except (KeyError, IndexError, TypeError):
        return nomes

    # Tolerancia proporcional ao tamanho da cena, para absorver arredondamento.
    if candidatos:
        extensao = max(
            max(c[1][1] - c[1][0], c[1][3] - c[1][2], c[1][5] - c[1][4])
            for c in candidatos
        )
        tolerancia = max(extensao * 0.02, 1e-4)
    else:
        return nomes

    usados = set()
    for indice, ator in enumerate(atores):
        limites = ator.GetBounds()
        melhor, menor_erro = None, float("inf")
        for posicao, (nome, caixa) in enumerate(candidatos):
            if posicao in usados or not nome:
                continue
            erro = max(abs(limites[i] - caixa[i]) for i in range(6))
            if erro < menor_erro:
                melhor, menor_erro = posicao, erro
        if melhor is not None and menor_erro <= tolerancia:
            usados.add(melhor)
            nomes[indice] = candidatos[melhor][0].replace("_", " ")
    return nomes


# ----------------------------------------------------------------------------
# Montagem da cena 3D
# ----------------------------------------------------------------------------
def carregar_modelo(janela: vtk.vtkRenderWindow, caminho: str):
    """Importa o .glb e devolve o renderizador e a lista (nome, ator) das partes."""
    importador = vtk.vtkGLTFImporter()
    importador.SetFileName(caminho)
    importador.SetRenderWindow(janela)
    importador.Update()

    renderizador = janela.GetRenderers().GetFirstRenderer()
    if renderizador is None or renderizador.GetActors().GetNumberOfItems() == 0:
        sys.exit(f"Nao consegui ler geometria de: {caminho}")

    colecao = renderizador.GetActors()
    colecao.InitTraversal()
    atores = [colecao.GetNextActor() for _ in range(colecao.GetNumberOfItems())]

    partes = list(zip(nomes_das_partes(caminho, atores), atores))
    return renderizador, partes


def preparar_renderizador(renderizador: vtk.vtkRenderer) -> None:
    """Fundo branco, meia tela e uma iluminacao decente."""
    renderizador.SetBackground(*COR_FUNDO)
    if LADO_DO_MODELO.lower().startswith("esq"):
        renderizador.SetViewport(0.0, 0.0, 0.5, 1.0)
    else:
        renderizador.SetViewport(0.5, 0.0, 1.0, 1.0)

    # Luz de estudio (chave + preenchimento + contraluz) no lugar da lanterna
    # padrao do VTK, que deixa o modelo chapado.
    renderizador.AutomaticLightCreationOff()
    renderizador.RemoveAllLights()
    luzes = vtk.vtkLightKit()
    luzes.SetKeyLightIntensity(1.1)
    luzes.SetKeyToFillRatio(2.5)
    luzes.SetKeyToHeadRatio(3.0)
    luzes.SetKeyToBackRatio(3.0)
    luzes.AddLightsToRenderer(renderizador)

    renderizador.ResetCamera()
    camera = renderizador.GetActiveCamera()
    camera.Azimuth(-ANGULO_INICIAL)
    camera.Elevation(ELEVACAO_CAMERA)
    camera.OrthogonalizeViewUp()
    renderizador.ResetCamera()
    camera.Zoom(ZOOM)


def criar_fundo(janela: vtk.vtkRenderWindow) -> vtk.vtkRenderer:
    """Renderizador vazio que pinta de branco a metade sem modelo."""
    fundo = vtk.vtkRenderer()
    fundo.SetBackground(*COR_FUNDO)
    fundo.InteractiveOff()
    if LADO_DO_MODELO.lower().startswith("esq"):
        fundo.SetViewport(0.5, 0.0, 1.0, 1.0)
    else:
        fundo.SetViewport(0.0, 0.0, 0.5, 1.0)

    if MOSTRAR_INSTRUCOES:
        texto = vtk.vtkTextActor()
        texto.SetInput(
            "Botão esquerdo: girar\n"
            "Roda do mouse: zoom\n"
            "Botão do meio: arrastar\n\n"
            "Setas: girar\n"
            "+ e -: zoom\n"
            "R: voltar ao início\n"
            "V: abrir o menu de partes\n"
            "S: ligar o slider de corte\n"
            "T: trip (barras caem)\n"
            "B: rearmar depois do trip\n"
            "A: retirar o banco L de uma vez (teste)\n"
            "I: reiniciar o reator\n"
            "J / K: diluir / borar\n"
            "X: acelerar a física\n"
            "Q: sair"
        )
        propriedade = texto.GetTextProperty()
        propriedade.SetFontFamilyToArial()
        propriedade.SetFontSize(16)
        propriedade.SetColor(0.35, 0.35, 0.38)
        propriedade.SetLineSpacing(1.3)
        texto.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        # Fica embaixo para o menu suspenso nao passar por cima.
        texto.SetPosition(0.10, 0.06)
        fundo.AddViewProp(texto)

    janela.AddRenderer(fundo)
    return fundo


# ----------------------------------------------------------------------------
# Interacao: rotacao livre + zoom
# ----------------------------------------------------------------------------
class NavegacaoLivre(vtk.vtkInteractorStyleTrackballCamera):
    """Estilo trackball (rotacao livre, zoom e arrasto) com atalhos e menu.

    O comportamento do mouse vem pronto do vtkInteractorStyleTrackballCamera:
        - botao esquerdo arrastando -> gira em qualquer direcao
        - roda do mouse            -> zoom
        - botao do meio arrastando -> desloca o modelo na tela
        - botao direito arrastando -> zoom continuo
    Aqui eu acrescento os atalhos de teclado e desvio o clique quando ele cai
    em cima do menu.
    """

    def __init__(self, renderizador: vtk.vtkRenderer, janela: vtk.vtkRenderWindow,
                 menu: MenuVisualizacao | None = None,
                 painel: PainelDeControle | None = None):
        self.renderizador = renderizador
        self.janela = janela
        self.menu = menu
        self.painel = painel
        self.barras = None                # ControleDeBarras, ligado depois

        self.SetMotionFactor(SENSIBILIDADE)
        self.SetMouseWheelMotionFactor(ZOOM_RODA)

        # Guarda o enquadramento inicial para a tecla R poder voltar a ele.
        self.camera_inicial = vtk.vtkCamera()
        self.camera_inicial.DeepCopy(renderizador.GetActiveCamera())

        # Com observadores registrados, o VTK deixa de aplicar os atalhos
        # padrao dele (w = wireframe, s = solido, etc.) para esses eventos.
        self.AddObserver("LeftButtonPressEvent", self._clique_esquerdo)
        self.AddObserver("LeftButtonReleaseEvent", self._soltou_esquerdo)
        self.AddObserver("MouseMoveEvent", self._moveu_mouse)
        self.AddObserver("KeyPressEvent", self._tecla_pressionada)
        self.AddObserver("CharEvent", self._caractere)

    # -- acoes ---------------------------------------------------------------
    def aplicar_zoom(self, fator: float) -> None:
        camera = self.renderizador.GetActiveCamera()
        camera.Dolly(fator)
        self.renderizador.ResetCameraClippingRange()
        self.janela.Render()

    def girar_camera(self, azimute: float, elevacao: float) -> None:
        camera = self.renderizador.GetActiveCamera()
        camera.Azimuth(azimute)
        camera.Elevation(elevacao)
        camera.OrthogonalizeViewUp()
        self.renderizador.ResetCameraClippingRange()
        self.janela.Render()

    def voltar_ao_inicio(self) -> None:
        self.renderizador.GetActiveCamera().DeepCopy(self.camera_inicial)
        self.renderizador.ResetCameraClippingRange()
        self.janela.Render()

    # -- mouse ---------------------------------------------------------------
    def _clique_esquerdo(self, obj, evento) -> None:
        x, y = self.GetInteractor().GetEventPosition()
        if self.menu is not None and self.menu.clique(x, y):
            return  # o clique era do menu: a camera nao se mexe
        if self.painel is not None and self.painel.clique(x, y):
            return  # segurou um botao do painel
        self.OnLeftButtonDown()  # comportamento normal de girar

    def _moveu_mouse(self, obj, evento) -> None:
        if self.painel is not None and self.painel.pressionando():
            x, y = self.GetInteractor().GetEventPosition()
            self.painel.mover_mouse(x, y)  # saiu de cima do botao = soltou
            return
        if self.menu is not None and self.menu.arrastando():
            x, y = self.GetInteractor().GetEventPosition()
            self.menu.arrastar(x, y)
            return  # puxando a regua: a camera fica parada
        self.OnMouseMove()

    def _soltou_esquerdo(self, obj, evento) -> None:
        if self.painel is not None and self.painel.soltar():
            return  # soltou o botao do painel: a barra para
        if self.menu is not None and self.menu.arrastando():
            self.menu.soltar()
            return
        self.OnLeftButtonUp()

    # -- teclado -------------------------------------------------------------
    def _tecla_pressionada(self, obj, evento) -> None:
        tecla = self.GetInteractor().GetKeySym()
        if tecla == "Left":
            self.girar_camera(-PASSO_TECLADO, 0.0)
        elif tecla == "Right":
            self.girar_camera(PASSO_TECLADO, 0.0)
        elif tecla == "Up":
            self.girar_camera(0.0, PASSO_TECLADO)
        elif tecla == "Down":
            self.girar_camera(0.0, -PASSO_TECLADO)

    def _caractere(self, obj, evento) -> None:
        tecla = (self.GetInteractor().GetKeySym() or "").lower()
        if tecla in ("plus", "equal", "kp_add"):
            self.aplicar_zoom(ZOOM_TECLADO)
        elif tecla in ("minus", "underscore", "kp_subtract"):
            self.aplicar_zoom(1.0 / ZOOM_TECLADO)
        elif tecla == "r":
            self.voltar_ao_inicio()
        elif tecla == "v" and self.menu is not None:
            self.menu.alternar_menu()
        elif tecla == "s" and self.menu is not None:
            self.menu.alternar_regua()
        elif tecla == "t" and self.barras is not None:
            self.barras.trip()
        elif tecla == "b" and self.barras is not None:
            self.barras.rearmar()
        elif tecla == "a" and self.barras is not None:
            self.barras.retirar_desligamento()
        elif tecla in ("q", "e", "escape"):
            self.GetInteractor().TerminateApp()


# ----------------------------------------------------------------------------
# Barras de controle
# ----------------------------------------------------------------------------
class BancoDeBarras:
    """Um banco de barras: todas sobem e descem juntas, passo a passo.

    `passos` segue a convencao dos contadores de passos: 0 e toda inserida e
    PASSOS_TOTAIS e toda retirada. O modelo abre com as barras retiradas, entao
    na posicao PASSOS_TOTAIS o deslocamento e zero e em 0 ele e `curso` para
    baixo. O curso e medido do proprio modelo (ver _medir_curso).

    O ator importado do glTF ja vem com escala embutida na matriz dele, entao
    SetPosition nao anda em unidades do mundo. Por isso eu guardo a parte
    linear da matriz original de cada ator, inverto, e converto o deslocamento
    desejado para o referencial dele antes de aplicar.
    """

    def __init__(self, linha: int, sigla: str, tipo: str, atores: list,
                 curso: float) -> None:
        self.linha = linha
        self.sigla = sigla
        self.tipo = tipo
        self.atores = atores
        self.curso = max(0.0, curso)
        self.passos = 0.0 if COMECAR_INSERIDAS else float(PASSOS_TOTAIS)
        self.sentido = 0                  # +1 retirando, -1 inserindo, 0 parado
        self.acumulado = 0.0              # fracao de passo ja "esperada"
        self.travado = False              # OUT recusado pelo procedimento/trip
        self.velocidade_de_queda = 0.0    # passos por segundo, durante o trip

        self.para_o_ator = []
        for ator in atores:
            matriz = vtk.vtkMatrix4x4()
            matriz.DeepCopy(ator.GetMatrix())
            for eixo in range(3):
                matriz.SetElement(eixo, 3, 0.0)  # so a parte linear interessa
            matriz.Invert()
            self.para_o_ator.append(matriz)
        self.aplicar()

    @property
    def inserida(self) -> bool:
        return self.passos <= 0.0

    @property
    def retirada(self) -> bool:
        return self.passos >= PASSOS_TOTAIS

    def andar(self, segundos: float) -> None:
        """Da os passos que couberem no tempo, na velocidade do tipo do banco."""
        if self.sentido == 0:
            return
        taxa = PASSOS_POR_MINUTO[self.tipo] / 60.0 * ACELERACAO_DO_TEMPO
        self.acumulado += taxa * segundos
        while self.acumulado >= 1.0 and self.sentido != 0:
            self.acumulado -= 1.0
            self.passos = min(PASSOS_TOTAIS, max(0, round(self.passos) + self.sentido))
            if (self.sentido > 0 and self.retirada) or (self.sentido < 0 and self.inserida):
                self.sentido = 0
        self.aplicar()

    def cair(self, segundos: float) -> None:
        """Queda livre ate o dashpot, depois descida freada ate o fundo."""
        if self.inserida:
            self.velocidade_de_queda = 0.0
            return
        inicio_do_dashpot = FRACAO_DO_DASHPOT * PASSOS_TOTAIS
        if self.passos > inicio_do_dashpot:
            percurso = PASSOS_TOTAIS - inicio_do_dashpot
            aceleracao = 2.0 * percurso / TEMPO_ATE_O_DASHPOT ** 2
            self.velocidade_de_queda += aceleracao * segundos
        else:
            self.velocidade_de_queda = inicio_do_dashpot / TEMPO_NO_DASHPOT
        self.passos = max(0.0, self.passos - self.velocidade_de_queda * segundos)
        self.aplicar()

    def aplicar(self) -> None:
        # Retirada total = posicao do modelo; cada passo a menos desce a barra.
        descida = self.curso * (1.0 - self.passos / PASSOS_TOTAIS)
        deslocamento = [-c * descida for c in DIRECAO_DA_PECA] + [0.0]
        for ator, matriz in zip(self.atores, self.para_o_ator):
            x, y, z, _ = matriz.MultiplyPoint(deslocamento)
            ator.SetPosition(x, y, z)


def _altura(limites: tuple, topo: bool) -> float:
    """Altura (ao longo de DIRECAO_DA_PECA) do topo ou do fundo de uma caixa."""
    cantos = [
        (limites[i & 1], limites[2 + ((i >> 1) & 1)], limites[4 + ((i >> 2) & 1)])
        for i in range(8)
    ]
    alturas = [sum(c * d for c, d in zip(canto, DIRECAO_DA_PECA)) for canto in cantos]
    return max(alturas) if topo else min(alturas)


def _se_sobrepoem(a: tuple, b: tuple) -> bool:
    """As duas caixas se cruzam no plano horizontal (X e Z)?"""
    return a[0] <= b[1] and b[0] <= a[1] and a[4] <= b[5] and b[4] <= a[5]


def dividir_em_bancos(partes: list) -> dict:
    """Distribui as barras do modelo pelos bancos de BANCOS.

    Barra, aqui, e todo Support que sobe acima da Top Platform (os Supports
    curtos, que ficam dentro do vaso, sao estrutura fixa). Cada barra vira um
    par de coordenadas inteiras na grade do nucleo, e o anel dela decide o
    banco. Devolve {sigla: [atores]}.
    """
    topo = None
    for p in partes:
        if p["nome"].strip().lower() == NOME_TOPO_DO_REATOR:
            topo = _altura(p["ator"].GetBounds(), topo=True)
    barras = []
    for p in partes:
        if not p["nome"].strip().lower().startswith(PREFIXO_DAS_BARRAS):
            continue
        limites = p["ator"].GetBounds()
        if topo is None or _altura(limites, topo=True) > topo:
            centro = ((limites[0] + limites[1]) / 2.0, (limites[4] + limites[5]) / 2.0)
            barras.append((p["ator"], centro))
    if not barras:
        return {}

    xs = [c[0] for _, c in barras]
    zs = [c[1] for _, c in barras]
    cx, cz = (min(xs) + max(xs)) / 2.0, (min(zs) + max(zs)) / 2.0
    # Passo da grade = menor distancia entre duas barras vizinhas.
    distancias = [
        max(abs(a[0] - b[0]), abs(a[1] - b[1]))
        for i, (_, a) in enumerate(barras) for _, b in barras[i + 1:]
    ]
    passo = min((d for d in distancias if d > 1e-6), default=1.0)

    anel_do_banco = {}
    for _, sigla, _, aneis in BANCOS:
        for anel in aneis:
            anel_do_banco[tuple(anel)] = sigla

    bancos = {sigla: [] for _, sigla, _, _ in BANCOS}
    sem_banco = 0
    for ator, (x, z) in barras:
        i, j = abs(round((x - cx) / passo)), abs(round((z - cz) / passo))
        sigla = anel_do_banco.get((min(i, j), max(i, j)))
        if sigla is None:
            sem_banco += 1
        else:
            bancos[sigla].append(ator)
    if sem_banco:
        print(f"Aviso: {sem_banco} barra(s) fora de qualquer banco ficam paradas.")
    return bancos


def _medir_curso(atores: list, partes: list) -> float:
    """Distancia que a barra desce ate ficar toda inserida.

    Com "fuel rods", usa o fundo das Wire Things que ficam na mesma coluna da
    barra (se nao achar nenhuma na coluna, usa a mais baixa de todas). Com
    "base interna", usa o topo da Internal Platform Bottom.
    """
    fundo_da_barra = min(_altura(a.GetBounds(), topo=False) for a in atores)
    caixa_da_barra = [float("inf"), float("-inf")] * 3
    for ator in atores:
        limites = ator.GetBounds()
        for e in range(3):
            caixa_da_barra[2 * e] = min(caixa_da_barra[2 * e], limites[2 * e])
            caixa_da_barra[2 * e + 1] = max(caixa_da_barra[2 * e + 1], limites[2 * e + 1])

    alvo = None
    if FIM_DA_INSERCAO.lower().startswith("fuel"):
        fuel = [p["ator"].GetBounds() for p in partes
                if p["nome"].strip().lower().startswith(PREFIXO_FUEL_RODS)]
        na_coluna = [f for f in fuel if _se_sobrepoem(f, caixa_da_barra)]
        escolhidas = na_coluna or fuel
        if escolhidas:
            alvo = min(_altura(f, topo=False) for f in escolhidas)
    if alvo is None:
        for p in partes:
            if p["nome"].strip().lower() == NOME_BASE_INTERNA:
                alvo = _altura(p["ator"].GetBounds(), topo=True)
    if alvo is None:
        print("Aviso: nao achei o referencial de insercao; a barra nao vai descer.")
        return 0.0
    return max(0.0, fundo_da_barra - alvo)


class ControleDeBarras:
    """Sistema de controle das barras: botoes, travas, trip, LEDs e contadores.

    - Segurar IN: o banco da linha desce passo a passo. OUT: sobe. Soltou, para.
    - O banco L anda a 64 passos/min; os grupos D, a 48.
    - Com EXIGIR_SEQUENCIA, um grupo D so sai com o L acima de
      FRACAO_MINIMA_DO_L e com os grupos D anteriores todos fora.
      Inserir nunca e bloqueado.
    - Tecla T: trip. Todas as barras caem por gravidade e a retirada fica
      bloqueada ate rearmar os disjuntores com a tecla B.
    - LED verde: banco todo inserido. Vermelho: todo retirado.
    - Display: sigla do banco e o contador de passos (000 a PASSOS_TOTAIS).
    """

    def __init__(self, partes: list, painel: PainelDeControle,
                 janela: vtk.vtkRenderWindow) -> None:
        self.painel = painel
        self.janela = janela
        self.interator = None
        self.id_temporizador = None
        self.instante_anterior = 0.0
        self.em_trip = False

        atores_por_banco = dividir_em_bancos(partes)
        self.bancos = []
        self.banco_da_linha = {}
        for linha, sigla, tipo, _ in BANCOS:
            atores = atores_por_banco.get(sigla, [])
            if not atores or not 1 <= linha <= len(painel.linhas):
                continue
            banco = BancoDeBarras(linha, sigla, tipo, atores,
                                  _medir_curso(atores, partes))
            self.bancos.append(banco)
            self.banco_da_linha[linha] = banco

        painel.ao_pressionar = self.pressionar
        painel.ao_soltar = self.soltar
        self.atualizar_indicadores(renderizar=False)

    def conectar(self, interator: vtk.vtkRenderWindowInteractor) -> None:
        self.interator = interator
        interator.AddObserver("TimerEvent", self._ao_temporizador)

    # -- regras --------------------------------------------------------------
    def _pode_retirar(self, banco: BancoDeBarras) -> bool:
        if self.em_trip:
            return False
        if EXIGIR_SEQUENCIA and banco.tipo == "controle":
            minimo = FRACAO_MINIMA_DO_L * PASSOS_TOTAIS
            if any(b.passos < minimo for b in self.bancos if b.tipo == "desligamento"):
                return False
            # Grupos D em serie: os das linhas de cima tem que estar todos fora.
            return all(b.retirada for b in self.bancos
                       if b.tipo == "controle" and b.linha < banco.linha)
        return True

    # -- botoes --------------------------------------------------------------
    def pressionar(self, numero: int, tipo: str) -> None:
        banco = self.banco_da_linha.get(numero)
        if banco is None or self.em_trip:
            if banco is not None and tipo == "out":
                banco.travado = True
                self.atualizar_indicadores()
            return
        if tipo == "out" and not self._pode_retirar(banco):
            banco.travado = True
            self.atualizar_indicadores()
            return
        banco.sentido = 1 if tipo == "out" else -1
        if (banco.sentido > 0 and banco.retirada) or (banco.sentido < 0 and banco.inserida):
            banco.sentido = 0
            return
        # O primeiro passo sai logo que o botao e apertado, como no painel real.
        banco.acumulado = 1.0
        banco.andar(0.0)
        self.atualizar_indicadores()
        self._iniciar()

    def soltar(self, numero: int, tipo: str) -> None:
        banco = self.banco_da_linha.get(numero)
        if banco is not None:
            banco.sentido = 0
            banco.acumulado = 0.0
            if banco.travado:
                banco.travado = False
                self.atualizar_indicadores()

    # -- trip ----------------------------------------------------------------
    def trip(self) -> None:
        """Abre os disjuntores de trip: as garras soltam e tudo cai."""
        if self.em_trip:
            return
        self.em_trip = True
        for banco in self.bancos:
            banco.sentido = 0
            banco.velocidade_de_queda = 0.0
        self.atualizar_indicadores()
        self._iniciar()

    def retirar_desligamento(self) -> None:
        """Atalho de teste: poe o banco L todo fora, na hora.

        Na usina ele sobe passo a passo como os outros; aqui pula direto
        para PASSOS_TOTAIS para nao ter que segurar OUT por minutos. Com os
        grupos D no fundo o reator continua subcritico.
        """
        if self.em_trip:
            return
        for banco in self.bancos:
            if banco.tipo == "desligamento":
                banco.sentido = 0
                banco.acumulado = 0.0
                banco.travado = False
                banco.passos = float(PASSOS_TOTAIS)
                banco.aplicar()
        self.atualizar_indicadores()

    def rearmar(self) -> None:
        """Fecha os disjuntores de novo (so depois que tudo chegou no fundo)."""
        if self.em_trip and all(b.inserida for b in self.bancos):
            self.em_trip = False
            self.atualizar_indicadores()

    # -- relogio -------------------------------------------------------------
    def _em_movimento(self) -> bool:
        if self.em_trip and not all(b.inserida for b in self.bancos):
            return True
        return any(b.sentido for b in self.bancos)

    def _iniciar(self) -> None:
        if self.interator is None or self.id_temporizador is not None:
            return
        self.instante_anterior = time.monotonic()
        self.id_temporizador = self.interator.CreateRepeatingTimer(INTERVALO_DA_BARRA)

    def _parar(self) -> None:
        if self.id_temporizador is not None and self.interator is not None:
            self.interator.DestroyTimer(self.id_temporizador)
        self.id_temporizador = None

    def _ao_temporizador(self, obj, evento) -> None:
        if self.id_temporizador is None:
            return
        # O quanto anda depende do tempo real que passou, e nao de quantos
        # eventos chegaram (o menu tambem usa o relogio do interator).
        agora = time.monotonic()
        segundos = min(0.1, agora - self.instante_anterior)
        self.instante_anterior = agora
        self.avancar(segundos)
        if not self._em_movimento():
            self._parar()

    def avancar(self, segundos: float) -> None:
        antes = [b.passos for b in self.bancos]
        for banco in self.bancos:
            if self.em_trip:
                banco.cair(segundos)
            else:
                banco.andar(segundos)
        # So redesenha quando algum banco de fato deu passo (ou esta caindo).
        if [b.passos for b in self.bancos] != antes:
            self.atualizar_indicadores()

    # -- LEDs e contadores ---------------------------------------------------
    def atualizar_indicadores(self, renderizar: bool = True) -> None:
        for banco in self.bancos:
            self.painel.definir_leds(
                banco.linha, banco.inserida, banco.retirada, renderizar=False
            )
            if self.em_trip:
                aviso = "TRIP"
            elif banco.travado:
                aviso = "RETIRADA BLOQUEADA"
            else:
                aviso = ""
            texto = f"{banco.sigla}   {int(round(banco.passos)):03d} passos"
            self.painel.linhas[banco.linha - 1].texto_display.SetInput(
                texto + ("\n" + aviso if aviso else "")
            )
        if renderizar:
            self.janela.Render()


# ----------------------------------------------------------------------------
# Programa
# ----------------------------------------------------------------------------
def _lado_vazio() -> str:
    """O painel de controle vai no lado da janela que o modelo nao ocupa."""
    return "direita" if LADO_DO_MODELO.lower().startswith("esq") else "esquerda"


def _identificar_no_windows() -> None:
    """No Windows, faz a barra de tarefas tratar o programa como "Core3D".

    Sem isso o Windows agrupa a janela com o python.exe e mostra o icone do
    Python na barra de tarefas, em vez do icone da janela.
    """
    if sys.platform.startswith("win"):
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(NOME_DO_PROGRAMA)
        except Exception:
            pass


def _carregar_icone():
    """Le o core3d.png da pasta do script (None se nao achar)."""
    caminho = os.path.join(os.path.dirname(os.path.abspath(__file__)), ARQUIVO_DO_ICONE)
    if not os.path.isfile(caminho):
        return None
    leitor = vtk.vtkPNGReader()
    leitor.SetFileName(caminho)
    leitor.Update()
    return leitor.GetOutput()


ARQUIVO_DO_ICONE_WINDOWS = "core3d.ico"   # na mesma pasta do script


def _hwnd_da_janela(janela: vtk.vtkRenderWindow) -> int:
    """Handle (HWND) da janela no Windows, ou 0 se ainda nao existe."""
    import ctypes
    import re
    ident = janela.GetGenericWindowId()
    if isinstance(ident, int):
        return ident
    if isinstance(ident, str):
        # O VTK devolve algo como "_000000000012a4b6_p_void".
        achado = re.search(r"_([0-9a-fA-F]+)_p_", ident)
        if achado:
            return int(achado.group(1), 16)
    return ctypes.windll.user32.FindWindowW(None, NOME_DO_PROGRAMA) or 0


def _icone_na_barra_do_windows(janela: vtk.vtkRenderWindow) -> None:
    """Troca o icone da janela e o da barra de tarefas pelo core3d.ico.

    O SetIcon do VTK nao muda o icone no Windows, entao isso usa a API do
    sistema: carrega o .ico em dois tamanhos (o pequeno vai na barra de
    titulo, o grande na barra de tarefas e no Alt+Tab) e manda para a janela.
    """
    import ctypes
    from ctypes import wintypes

    caminho = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           ARQUIVO_DO_ICONE_WINDOWS)
    hwnd = _hwnd_da_janela(janela)
    if not hwnd or not os.path.isfile(caminho):
        return

    user32 = ctypes.windll.user32
    user32.LoadImageW.restype = wintypes.HANDLE
    user32.LoadImageW.argtypes = (wintypes.HINSTANCE, wintypes.LPCWSTR, wintypes.UINT,
                                  ctypes.c_int, ctypes.c_int, wintypes.UINT)
    user32.SendMessageW.restype = ctypes.c_ssize_t
    user32.SendMessageW.argtypes = (wintypes.HWND, wintypes.UINT,
                                    ctypes.c_size_t, ctypes.c_ssize_t)
    IMAGE_ICON, LR_LOADFROMFILE = 1, 0x10
    WM_SETICON, ICON_SMALL, ICON_BIG = 0x0080, 0, 1
    SM_CXICON, SM_CXSMICON = 11, 49

    grande = user32.GetSystemMetrics(SM_CXICON) * 2      # nitido em telas com zoom
    pequeno = user32.GetSystemMetrics(SM_CXSMICON) * 2
    h_grande = user32.LoadImageW(None, caminho, IMAGE_ICON, grande, grande, LR_LOADFROMFILE)
    h_pequeno = user32.LoadImageW(None, caminho, IMAGE_ICON, pequeno, pequeno, LR_LOADFROMFILE)
    if h_grande:
        user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG, h_grande)
    if h_pequeno:
        user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, h_pequeno)
    # Guarda os handles: se forem liberados, o icone some.
    janela._icones_windows = (h_grande, h_pequeno)


def aplicar_icone(janela: vtk.vtkRenderWindow) -> None:
    """Poe o icone na janela (chamado de novo depois que ela existe na tela)."""
    if sys.platform.startswith("win"):
        try:
            _icone_na_barra_do_windows(janela)
        except Exception as erro:
            print(f"Aviso: nao consegui trocar o icone ({erro}).")
        return
    icone = getattr(janela, "_icone_core3d", None)
    if icone is not None and hasattr(janela, "SetIcon"):
        janela.SetIcon(icone)


def montar(caminho: str, offscreen: bool = False):
    _identificar_no_windows()
    janela = vtk.vtkRenderWindow()
    janela.SetSize(LARGURA, ALTURA)
    janela.SetWindowName(NOME_DO_PROGRAMA)      # so o nome, sem o caminho do modelo
    janela._icone_core3d = _carregar_icone()
    aplicar_icone(janela)
    janela.SetMultiSamples(8)  # antialiasing
    if offscreen:
        janela.SetOffScreenRendering(1)

    renderizador, partes = carregar_modelo(janela, caminho)
    preparar_renderizador(renderizador)
    criar_fundo(janela)
    # O painel entra antes do menu para o dropdown sempre passar por cima dele.
    painel = PainelDeControle(janela, lado=_lado_vazio())
    menu = MenuVisualizacao(janela, renderizador, partes)
    return janela, renderizador, menu, painel


def main() -> None:
    caminho = localizar_modelo(sys.argv[1:])
    janela, renderizador, menu, painel = montar(caminho)

    interator = vtk.vtkRenderWindowInteractor()
    interator.SetRenderWindow(janela)
    estilo = NavegacaoLivre(renderizador, janela, menu, painel)
    interator.SetInteractorStyle(estilo)
    menu.conectar(interator)

    # Cada linha do painel move um banco (IN insere, OUT retira).
    # T = trip (todas as barras caem); B = rearmar depois do trip.
    # A = atalho de teste: retira o banco L de uma vez.
    barras = ControleDeBarras(menu.partes, painel, janela)
    barras.conectar(interator)
    estilo.barras = barras

    # Fisica do nucleo, so na regiao dos fuel rods (nucleo.py).
    # J = diluir boro, K = borar, X = acelerar o tempo da fisica.
    # Os quadrantes so sao mostrados (sem selecao nem comando por quadrante).
    nucleo = MonitorDoNucleo(menu.partes, barras, painel, janela, renderizador)
    nucleo.conectar(interator)

    interator.Initialize()
    janela.Render()
    # Alguns sistemas so aceitam o icone com a janela ja aberta.
    aplicar_icone(janela)
    janela.SetWindowName(NOME_DO_PROGRAMA)
    interator.Start()


if __name__ == "__main__":
    main()
