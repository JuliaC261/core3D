#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualizador 3D para modelos .glb (glTF binario).

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

import vtk

from dropdown import MenuVisualizacao
from painel import PainelDeControle

# ----------------------------------------------------------------------------
# Configuracoes gerais
# ----------------------------------------------------------------------------
ARQUIVO_PADRAO = "Yotuber_Nuclear_Reacts_To_Demon_Core.glb"

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
            return  # apertou um botao do painel
        self.OnLeftButtonDown()  # comportamento normal de girar

    def _moveu_mouse(self, obj, evento) -> None:
        if self.menu is not None and self.menu.arrastando():
            x, y = self.GetInteractor().GetEventPosition()
            self.menu.arrastar(x, y)
            return  # puxando a regua: a camera fica parada
        self.OnMouseMove()

    def _soltou_esquerdo(self, obj, evento) -> None:
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
        elif tecla in ("q", "e", "escape"):
            self.GetInteractor().TerminateApp()


# ----------------------------------------------------------------------------
# Programa
# ----------------------------------------------------------------------------
def _lado_vazio() -> str:
    """O painel de controle vai no lado da janela que o modelo nao ocupa."""
    return "direita" if LADO_DO_MODELO.lower().startswith("esq") else "esquerda"


def montar(caminho: str, offscreen: bool = False):
    janela = vtk.vtkRenderWindow()
    janela.SetSize(LARGURA, ALTURA)
    janela.SetWindowName("Visualizador 3D - " + os.path.basename(caminho))
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
    interator.SetInteractorStyle(
        NavegacaoLivre(renderizador, janela, menu, painel)
    )
    menu.conectar(interator)

    interator.Initialize()
    janela.Render()
    interator.Start()


if __name__ == "__main__":
    main()