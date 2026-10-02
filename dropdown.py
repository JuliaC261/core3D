#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Menu suspenso "Visualizacao" para o visualizador 3D.

Este arquivo cuida de tudo que aparece na barra do topo: o desenho em 2D, o
posicionamento das partes em linhas horizontais, a animacao de descida e o
liga/desliga de cada peca do modelo.

Quem usa isso e o visualizador_3d.py, mais ou menos assim:

    from menu_visualizacao import MenuVisualizacao

    menu = MenuVisualizacao(janela, partes)   # partes = [(nome, ator), ...]
    menu.conectar(interator)                  # da o relogio da animacao
    ...
    if menu.clique(x, y):                     # True = o menu consumiu o clique
        return
"""

from __future__ import annotations

import math
import time

import vtk

# ----------------------------------------------------------------------------
# Aparencia do menu
# ----------------------------------------------------------------------------
TITULO_MENU = "Visualização"
ALTURA_BARRA = 36                    # altura da faixa no topo, em pixels
ALTURA_ITEM = 28                     # altura de cada parte na lista
LARGURA_MINIMA_ITEM = 90             # largura minima de cada parte
ESPACO_ITEM = 4                      # respiro entre os itens
MARGEM_PAINEL = 6                    # respiro interno do painel
MARGEM_ESQUERDA = 10                 # distancia do menu ate a borda da janela
FONTE_MENU = 16
FONTE_ITEM = 14
TITULO_SLIDER = "Slider"             # segunda opcao da barra do topo
ANIMAR = True                        # painel desce deslizando ao abrir
DURACAO_ANIMACAO = 0.22              # duracao da animacao, em segundos
INTERVALO_ANIMACAO = 16              # intervalo entre quadros, em milissegundos

COR_BARRA = (1.00, 1.00, 1.00)
COR_LINHA = (0.85, 0.86, 0.88)
COR_TEXTO_MENU = (0.20, 0.22, 0.25)
COR_BOTAO_ABERTO = (0.93, 0.94, 0.96)
COR_PAINEL = (1.00, 1.00, 1.00)
COR_ITEM_LIGADO = (0.85, 0.94, 0.99)      # azul claro = parte visivel
COR_CONTORNO_LIGADO = (0.55, 0.78, 0.92)
COR_TEXTO_LIGADO = (0.06, 0.28, 0.42)
COR_ITEM_DESLIGADO = (0.97, 0.97, 0.97)   # cinza apagado = parte escondida
COR_TEXTO_DESLIGADO = (0.58, 0.60, 0.63)

# Regua de corte (o "slider")
COR_REGUA = (0.70, 0.22, 0.22)
ESPESSURA_REGUA = 1.5
MEIA_BASE_SETA = 13                  # metade da base do triangulo, em pixels
ALTURA_SETA = 16                     # altura do triangulo que serve de pegador
FOLGA_PEGADOR = 6                    # tolerancia extra para agarrar a regua
LADO_CORTADO = "direita"             # lado que some: "direita" ou "esquerda"
EIXO_DO_CORTE = "x"                  # eixo do modelo em que a regua corre


# ----------------------------------------------------------------------------
# Pecas graficas do menu
# ----------------------------------------------------------------------------
class Retangulo2D:
    """Retangulo desenhado em pixels na frente da cena (fundo + contorno)."""

    def __init__(self) -> None:
        self.pontos = vtk.vtkPoints()
        for _ in range(4):
            self.pontos.InsertNextPoint(0.0, 0.0, 0.0)

        self.area = (0, 0, 0, 0)
        self.visivel = True
        self.tem_contorno = False

        preenchimento = vtk.vtkPolyData()
        preenchimento.SetPoints(self.pontos)
        faces = vtk.vtkCellArray()
        faces.InsertNextCell(4)
        for i in range(4):
            faces.InsertCellPoint(i)
        preenchimento.SetPolys(faces)

        contorno = vtk.vtkPolyData()
        contorno.SetPoints(self.pontos)
        linhas = vtk.vtkCellArray()
        linhas.InsertNextCell(5)
        for i in (0, 1, 2, 3, 0):
            linhas.InsertCellPoint(i)
        contorno.SetLines(linhas)

        self.ator = self._criar_ator(preenchimento)
        self.ator_contorno = self._criar_ator(contorno)
        self.ator_contorno.GetProperty().SetLineWidth(1.0)

    @staticmethod
    def _criar_ator(poligonos: vtk.vtkPolyData) -> vtk.vtkActor2D:
        mapeador = vtk.vtkPolyDataMapper2D()
        mapeador.SetInputData(poligonos)
        coordenada = vtk.vtkCoordinate()
        coordenada.SetCoordinateSystemToDisplay()
        mapeador.SetTransformCoordinate(coordenada)
        ator = vtk.vtkActor2D()
        ator.SetMapper(mapeador)
        return ator

    def definir_area(self, x0: float, y0: float, x1: float, y1: float) -> None:
        self.area = (x0, y0, x1, y1)
        self.pontos.SetPoint(0, x0, y0, 0.0)
        self.pontos.SetPoint(1, x1, y0, 0.0)
        self.pontos.SetPoint(2, x1, y1, 0.0)
        self.pontos.SetPoint(3, x0, y1, 0.0)
        self.pontos.Modified()

    def contem(self, x: float, y: float) -> bool:
        x0, y0, x1, y1 = self.area
        return x0 <= x <= x1 and y0 <= y <= y1

    def pintar(self, cor: tuple, cor_do_contorno: tuple | None = None) -> None:
        """Troca as cores. Nao mexe em quem esta visivel ou nao."""
        self.ator.GetProperty().SetColor(*cor)
        self.tem_contorno = cor_do_contorno is not None
        if self.tem_contorno:
            self.ator_contorno.GetProperty().SetColor(*cor_do_contorno)
        self._aplicar_visibilidade()

    def mostrar(self, visivel: bool) -> None:
        self.visivel = bool(visivel)
        self._aplicar_visibilidade()

    def _aplicar_visibilidade(self) -> None:
        self.ator.SetVisibility(self.visivel)
        self.ator_contorno.SetVisibility(self.visivel and self.tem_contorno)

    def definir_camada(self, numero: int) -> None:
        self.ator.SetLayerNumber(numero)
        self.ator_contorno.SetLayerNumber(numero)

    def adicionar_em(self, renderizador: vtk.vtkRenderer) -> None:
        renderizador.AddViewProp(self.ator)
        renderizador.AddViewProp(self.ator_contorno)


def _criar_texto(conteudo: str, tamanho: int, cor: tuple) -> vtk.vtkTextActor:
    texto = vtk.vtkTextActor()
    texto.SetInput(conteudo)
    propriedade = texto.GetTextProperty()
    propriedade.SetFontFamilyToArial()
    propriedade.SetFontSize(tamanho)
    propriedade.SetColor(*cor)
    propriedade.SetJustificationToLeft()
    propriedade.SetVerticalJustificationToCentered()
    return texto


def _largura_do_texto(conteudo: str, tamanho: int) -> int:
    """Estimativa simples da largura em pixels (Arial tem ~0,58 de proporcao)."""
    return int(len(conteudo) * tamanho * 0.58)


# ----------------------------------------------------------------------------
# Regua de corte
# ----------------------------------------------------------------------------
def _criar_ator_2d(poligonos: vtk.vtkPolyData) -> vtk.vtkActor2D:
    """Ator desenhado direto em pixels da tela."""
    mapeador = vtk.vtkPolyDataMapper2D()
    mapeador.SetInputData(poligonos)
    coordenada = vtk.vtkCoordinate()
    coordenada.SetCoordinateSystemToDisplay()
    mapeador.SetTransformCoordinate(coordenada)
    ator = vtk.vtkActor2D()
    ator.SetMapper(mapeador)
    return ator


def _normalizar(vetor) -> tuple:
    tamanho = math.sqrt(sum(c * c for c in vetor))
    if tamanho == 0.0:
        return (0.0, 0.0, 0.0)
    return tuple(c / tamanho for c in vetor)


def _produto_vetorial(a, b) -> tuple:
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


class ReguaDeCorte:
    """Linha que fatia o modelo: o que fica de um lado dela some.

    O desenho e so uma linha em 2D com um triangulo de pegador. Quem corta de
    verdade e um vtkPlane ligado ao mapeador de cada peca: na hora de desenhar,
    o VTK descarta tudo que esta do lado negativo do plano.

    O plano fica preso ao MODELO (corta sempre ao longo do mesmo eixo dele), e
    nao a tela. E isso que abre a janela para dentro do reator: girando a peca
    depois do corte, voce olha para dentro dele em vez de ver uma parede lisa.
    A linha na tela e a projecao de onde o plano esta, recalculada a cada
    quadro, entao ela acompanha o modelo quando a camera gira.
    """

    def __init__(self, janela: vtk.vtkRenderWindow,
                 renderizador_3d: vtk.vtkRenderer,
                 renderizador_2d: vtk.vtkRenderer,
                 atores: list) -> None:
        self.janela = janela
        self.renderizador_3d = renderizador_3d
        self.atores = atores

        self.plano = vtk.vtkPlane()
        self.ativa = False
        self.arrastando = False

        self.deslocamento = 0.0            # posicao do corte, em unidades do modelo
        self.centro = (0.0, 0.0, 0.0)
        self.metade = 1.0
        self.eixo = (1.0, 0.0, 0.0)

        self.limites = (0.0, 100.0)        # faixa de pixels do visor 3D
        self.base = 0.0
        self.topo = 100.0
        self.x_linha = 0.0
        self.area_pegador = (0.0, 0.0, 0.0, 0.0)

        self.pontos_linha = vtk.vtkPoints()
        self.pontos_linha.InsertNextPoint(0.0, 0.0, 0.0)
        self.pontos_linha.InsertNextPoint(0.0, 0.0, 0.0)
        linha = vtk.vtkPolyData()
        linha.SetPoints(self.pontos_linha)
        celulas = vtk.vtkCellArray()
        celulas.InsertNextCell(2)
        celulas.InsertCellPoint(0)
        celulas.InsertCellPoint(1)
        linha.SetLines(celulas)
        self.ator_linha = _criar_ator_2d(linha)
        self.ator_linha.GetProperty().SetColor(*COR_REGUA)
        self.ator_linha.GetProperty().SetLineWidth(ESPESSURA_REGUA)
        self.ator_linha.SetLayerNumber(0)

        self.pontos_seta = vtk.vtkPoints()
        for _ in range(3):
            self.pontos_seta.InsertNextPoint(0.0, 0.0, 0.0)
        seta = vtk.vtkPolyData()
        seta.SetPoints(self.pontos_seta)
        faces = vtk.vtkCellArray()
        faces.InsertNextCell(3)
        for i in range(3):
            faces.InsertCellPoint(i)
        seta.SetPolys(faces)
        self.ator_seta = _criar_ator_2d(seta)
        self.ator_seta.GetProperty().SetColor(*COR_REGUA)
        self.ator_seta.SetLayerNumber(0)

        renderizador_2d.AddViewProp(self.ator_linha)
        renderizador_2d.AddViewProp(self.ator_seta)
        self._mostrar(False)

        self.medir()

    # -- medidas do modelo ---------------------------------------------------
    def medir(self) -> None:
        """Descobre o centro e a largura do modelo ao longo do eixo do corte."""
        caixa = [float("inf"), float("-inf")] * 3
        for ator in self.atores:
            limites = ator.GetBounds()
            for eixo in range(3):
                caixa[eixo * 2] = min(caixa[eixo * 2], limites[eixo * 2])
                caixa[eixo * 2 + 1] = max(caixa[eixo * 2 + 1], limites[eixo * 2 + 1])
        if caixa[0] == float("inf"):
            return

        self.centro = tuple(
            (caixa[eixo * 2] + caixa[eixo * 2 + 1]) / 2.0 for eixo in range(3)
        )
        indice = {"x": 0, "y": 1, "z": 2}.get(EIXO_DO_CORTE.lower(), 0)
        self.eixo = tuple(1.0 if e == indice else 0.0 for e in range(3))
        largura = caixa[indice * 2 + 1] - caixa[indice * 2]
        self.metade = max(largura / 2.0, 1e-6) * 1.05

    # -- posicionamento ------------------------------------------------------
    def dispor(self, largura: int, altura: int, base_da_barra: int) -> None:
        """Guarda a area util do visor 3D e redesenha a regua."""
        x0, y0, x1, y1 = self.renderizador_3d.GetViewport()
        self.limites = (x0 * largura, x1 * largura)
        self.topo = base_da_barra - ALTURA_SETA - 4
        self.base = y0 * altura
        self.atualizar()

    def _projetar(self, ponto) -> tuple:
        """Leva um ponto do mundo 3D para pixels da tela."""
        self.renderizador_3d.SetWorldPoint(ponto[0], ponto[1], ponto[2], 1.0)
        self.renderizador_3d.WorldToDisplay()
        return self.renderizador_3d.GetDisplayPoint()

    def _referencia(self) -> tuple:
        """Devolve (x do centro em pixels, quantos pixels vale 1 unidade)."""
        origem = self._projetar(self.centro)
        adiante = self._projetar(
            [c + e for c, e in zip(self.centro, self.eixo)]
        )
        return origem[0], adiante[0] - origem[0]

    def posicao_x(self) -> float:
        return self.x_linha

    def atualizar(self) -> None:
        """Refaz o plano e a linha. Chamado a cada quadro enquanto esta ativa."""
        if not self.ativa:
            return
        self.atualizar_plano()
        self._posicionar()

    def _posicionar(self) -> None:
        x_centro, escala = self._referencia()
        x = x_centro + escala * self.deslocamento
        esquerda, direita = self.limites
        self.x_linha = min(direita, max(esquerda, x))
        x = self.x_linha

        self.pontos_linha.SetPoint(0, x, self.base, 0.0)
        self.pontos_linha.SetPoint(1, x, self.topo + ALTURA_SETA, 0.0)
        self.pontos_linha.Modified()

        # Triangulo de ponta para baixo, apoiado no alto da linha.
        self.pontos_seta.SetPoint(0, x - MEIA_BASE_SETA, self.topo + ALTURA_SETA, 0.0)
        self.pontos_seta.SetPoint(1, x + MEIA_BASE_SETA, self.topo + ALTURA_SETA, 0.0)
        self.pontos_seta.SetPoint(2, x, self.topo, 0.0)
        self.pontos_seta.Modified()

        self.area_pegador = (
            x - MEIA_BASE_SETA - FOLGA_PEGADOR, self.topo - FOLGA_PEGADOR,
            x + MEIA_BASE_SETA + FOLGA_PEGADOR,
            self.topo + ALTURA_SETA + FOLGA_PEGADOR,
        )

    # -- o corte propriamente dito -------------------------------------------
    def atualizar_plano(self) -> None:
        origem = [
            c + e * self.deslocamento for c, e in zip(self.centro, self.eixo)
        ]
        sinal = -1.0 if LADO_CORTADO.lower().startswith("dir") else 1.0
        self.plano.SetOrigin(*origem)
        self.plano.SetNormal(*[c * sinal for c in self.eixo])

    def alternar(self) -> None:
        self.ativa = not self.ativa
        for ator in self.atores:
            mapeador = ator.GetMapper()
            if mapeador is None:
                continue
            if self.ativa:
                mapeador.AddClippingPlane(self.plano)
            else:
                mapeador.RemoveClippingPlane(self.plano)
        self._mostrar(self.ativa)
        self.atualizar()
        self.janela.Render()

    def _mostrar(self, visivel: bool) -> None:
        self.ator_linha.SetVisibility(visivel)
        self.ator_seta.SetVisibility(visivel)

    # -- arrastar ------------------------------------------------------------
    def pegar(self, x: float, y: float) -> bool:
        """Comeca a arrastar se o clique caiu no pegador ou em cima da linha."""
        if not self.ativa:
            return False

        x0, y0, x1, y1 = self.area_pegador
        no_pegador = x0 <= x <= x1 and y0 <= y <= y1
        na_linha = (
            abs(x - self.x_linha) <= FOLGA_PEGADOR
            and self.base <= y <= self.topo + ALTURA_SETA
        )
        if no_pegador or na_linha:
            self.arrastando = True
            self.arrastar(x, y)
            return True
        return False

    def arrastar(self, x: float, y: float) -> None:
        x_centro, escala = self._referencia()
        if abs(escala) < 0.5:
            return  # olhando o modelo de topo do eixo: arrastar nao faz sentido
        deslocamento = (x - x_centro) / escala
        self.deslocamento = min(self.metade, max(-self.metade, deslocamento))
        self.atualizar()
        self.janela.Render()

    def soltar(self) -> None:
        self.arrastando = False


# ----------------------------------------------------------------------------
# Menu "Visualizacao"
# ----------------------------------------------------------------------------
class MenuVisualizacao:
    """Barra no topo com o painel de partes, que desce em linhas horizontais."""

    def __init__(self, janela: vtk.vtkRenderWindow,
                 renderizador_3d: vtk.vtkRenderer, partes: list) -> None:
        self.janela = janela
        self.renderizador_3d = renderizador_3d
        self.partes = [
            {"nome": nome, "ator": ator, "visivel": True} for nome, ator in partes
        ]
        self.aberto = False
        self.tamanho_desenhado = (0, 0)

        # Animacao: 0 = escondido atras da barra, 1 = totalmente aberto.
        self.progresso = 0.0
        self.destino = 0.0
        self.altura_painel = 0
        self.interator = None
        self.id_temporizador = None
        self.instante_animacao = 0.0

        # Camada por cima da cena 3D: o VTK desenha as camadas em ordem.
        janela.SetNumberOfLayers(2)
        self.renderizador = vtk.vtkRenderer()
        self.renderizador.SetLayer(1)
        self.renderizador.InteractiveOff()
        janela.AddRenderer(self.renderizador)

        # O painel e os itens entram primeiro e ficam na camada de baixo, para
        # a barra do topo passar por cima deles enquanto o menu desliza.
        self.painel = Retangulo2D()
        self.painel.pintar(COR_PAINEL, COR_LINHA)
        self.painel.definir_camada(0)
        self.painel.adicionar_em(self.renderizador)

        self.itens = []
        for parte in self.partes:
            caixa = Retangulo2D()
            caixa.definir_camada(1)
            caixa.adicionar_em(self.renderizador)
            rotulo = _criar_texto(parte["nome"], FONTE_ITEM, COR_TEXTO_LIGADO)
            rotulo.SetLayerNumber(2)
            self.renderizador.AddViewProp(rotulo)
            self.itens.append({"caixa": caixa, "rotulo": rotulo})

        self.barra = Retangulo2D()
        self.barra.pintar(COR_BARRA, COR_LINHA)
        self.barra.definir_camada(3)
        self.barra.adicionar_em(self.renderizador)

        self.botao = Retangulo2D()
        self.botao.pintar(COR_BARRA)
        self.botao.definir_camada(4)
        self.botao.adicionar_em(self.renderizador)

        self.titulo = _criar_texto(TITULO_MENU, FONTE_MENU, COR_TEXTO_MENU)
        self.titulo.SetLayerNumber(5)
        self.renderizador.AddViewProp(self.titulo)

        # Segunda opcao da barra: liga e desliga a regua de corte.
        self.botao_slider = Retangulo2D()
        self.botao_slider.pintar(COR_BARRA)
        self.botao_slider.definir_camada(4)
        self.botao_slider.adicionar_em(self.renderizador)

        self.titulo_slider = _criar_texto(TITULO_SLIDER, FONTE_MENU, COR_TEXTO_MENU)
        self.titulo_slider.SetLayerNumber(5)
        self.renderizador.AddViewProp(self.titulo_slider)

        self.regua = ReguaDeCorte(
            janela, renderizador_3d, self.renderizador,
            [parte["ator"] for parte in self.partes],
        )

        self.dispor()
        self.atualizar_cores()
        self._mostrar_painel(False)

        # Refaz o posicionamento sempre que a janela mudar de tamanho.
        janela.AddObserver("StartEvent", self._antes_de_renderizar)

    def conectar(self, interator: vtk.vtkRenderWindowInteractor) -> None:
        """Liga o menu ao interator, que e quem dispara o relogio da animacao."""
        self.interator = interator
        interator.AddObserver("TimerEvent", self._ao_temporizador)

    # -- posicionamento ------------------------------------------------------
    def _antes_de_renderizar(self, obj, evento) -> None:
        if self.janela.GetSize() != self.tamanho_desenhado:
            self.dispor()
        # A linha acompanha o modelo, entao e reprojetada a cada quadro.
        self.regua.atualizar()

    def _suavizado(self) -> float:
        """Curva de desaceleracao: o painel chega devagar no fim."""
        return 1.0 - (1.0 - self.progresso) ** 3

    def dispor(self) -> None:
        largura, altura = self.janela.GetSize()
        self.tamanho_desenhado = (largura, altura)

        base_da_barra = altura - ALTURA_BARRA
        self.barra.definir_area(-2, base_da_barra, largura + 2, altura + 2)

        largura_botao = _largura_do_texto(TITULO_MENU, FONTE_MENU) + 24
        self.botao.definir_area(
            MARGEM_ESQUERDA, base_da_barra + 4,
            MARGEM_ESQUERDA + largura_botao, altura - 4,
        )
        self.titulo.SetDisplayPosition(
            int(MARGEM_ESQUERDA + 12), int(base_da_barra + ALTURA_BARRA // 2)
        )

        largura_slider = _largura_do_texto(TITULO_SLIDER, FONTE_MENU) + 24
        inicio_slider = MARGEM_ESQUERDA + largura_botao + 8
        self.botao_slider.definir_area(
            inicio_slider, base_da_barra + 4,
            inicio_slider + largura_slider, altura - 4,
        )
        self.titulo_slider.SetDisplayPosition(
            int(inicio_slider + 12), int(base_da_barra + ALTURA_BARRA // 2)
        )

        self._dispor_painel(largura, base_da_barra)

        # Com o painel aberto, o pegador da regua desce para nao ficar embaixo
        # dos chips das partes (e desliza junto durante a animacao).
        teto_da_regua = base_da_barra
        if self.progresso > 0.0:
            teto_da_regua = min(base_da_barra, self.painel.area[1] - 2)
        self.regua.dispor(largura, altura, teto_da_regua)

    def _dispor_painel(self, largura: int, base_da_barra: int) -> None:
        """Distribui as partes em linhas horizontais, quebrando quando lota."""
        disponivel = max(240, largura - 2 * (MARGEM_ESQUERDA + MARGEM_PAINEL))
        larguras = [
            max(LARGURA_MINIMA_ITEM, _largura_do_texto(p["nome"], FONTE_ITEM) + 28)
            for p in self.partes
        ]

        linhas, linha_atual, usado = [], [], 0
        for posicao, largura_item in enumerate(larguras):
            if linha_atual and usado + largura_item > disponivel:
                linhas.append(linha_atual)
                linha_atual, usado = [], 0
            linha_atual.append((posicao, largura_item))
            usado += largura_item + ESPACO_ITEM
        if linha_atual:
            linhas.append(linha_atual)
        if not linhas:
            return

        largura_conteudo = max(
            sum(w + ESPACO_ITEM for _, w in linha) - ESPACO_ITEM for linha in linhas
        )
        self.altura_painel = (
            2 * MARGEM_PAINEL
            + len(linhas) * ALTURA_ITEM
            + (len(linhas) - 1) * ESPACO_ITEM
        )

        # Enquanto a animacao roda, o painel fica deslocado para cima e some
        # atras da barra do topo.
        topo = base_da_barra + (1.0 - self._suavizado()) * self.altura_painel
        self.painel.definir_area(
            MARGEM_ESQUERDA, topo - self.altura_painel,
            MARGEM_ESQUERDA + largura_conteudo + 2 * MARGEM_PAINEL, topo,
        )

        for numero_da_linha, linha in enumerate(linhas):
            topo_da_linha = (
                topo - MARGEM_PAINEL
                - numero_da_linha * (ALTURA_ITEM + ESPACO_ITEM)
            )
            x = MARGEM_ESQUERDA + MARGEM_PAINEL
            for posicao, largura_item in linha:
                item = self.itens[posicao]
                item["caixa"].definir_area(
                    x, topo_da_linha - ALTURA_ITEM, x + largura_item, topo_da_linha
                )
                item["rotulo"].SetDisplayPosition(
                    int(x + 14), int(topo_da_linha - ALTURA_ITEM // 2)
                )
                x += largura_item + ESPACO_ITEM

    # -- aparencia -----------------------------------------------------------
    def atualizar_cores(self) -> None:
        for parte, item in zip(self.partes, self.itens):
            if parte["visivel"]:
                item["caixa"].pintar(COR_ITEM_LIGADO, COR_CONTORNO_LIGADO)
                item["rotulo"].GetTextProperty().SetColor(*COR_TEXTO_LIGADO)
            else:
                item["caixa"].pintar(COR_ITEM_DESLIGADO, COR_LINHA)
                item["rotulo"].GetTextProperty().SetColor(*COR_TEXTO_DESLIGADO)
        self.botao.pintar(COR_BOTAO_ABERTO if self.aberto else COR_BARRA)

        if self.regua.ativa:
            self.botao_slider.pintar(COR_ITEM_LIGADO, COR_CONTORNO_LIGADO)
            self.titulo_slider.GetTextProperty().SetColor(*COR_TEXTO_LIGADO)
        else:
            self.botao_slider.pintar(COR_BARRA)
            self.titulo_slider.GetTextProperty().SetColor(*COR_TEXTO_MENU)

    def _mostrar_painel(self, visivel: bool) -> None:
        self.painel.mostrar(visivel)
        for item in self.itens:
            item["caixa"].mostrar(visivel)
            item["rotulo"].SetVisibility(visivel)

    # -- animacao ------------------------------------------------------------
    def _iniciar_animacao(self) -> None:
        if not ANIMAR or self.interator is None:
            self.progresso = self.destino
            self._terminar_animacao()
            return
        self.instante_animacao = time.monotonic()
        if self.id_temporizador is None:
            self.id_temporizador = self.interator.CreateRepeatingTimer(
                INTERVALO_ANIMACAO
            )

    def _ao_temporizador(self, obj, evento) -> None:
        if self.id_temporizador is None:
            return
        # O quanto anda depende do tempo que passou, e nao de quantos eventos
        # chegaram: os timers das barras e da fisica tambem geram TimerEvent.
        agora = time.monotonic()
        passo = (agora - self.instante_animacao) / max(1e-3, DURACAO_ANIMACAO)
        self.instante_animacao = agora
        if self.destino > self.progresso:
            self.progresso = min(self.destino, self.progresso + passo)
        else:
            self.progresso = max(self.destino, self.progresso - passo)

        self.dispor()
        self.janela.Render()

        if self.progresso == self.destino:
            self.interator.DestroyTimer(self.id_temporizador)
            self.id_temporizador = None
            self._terminar_animacao()

    def _terminar_animacao(self) -> None:
        if self.progresso <= 0.0:
            self._mostrar_painel(False)
        self.dispor()
        self.janela.Render()

    # -- acoes ---------------------------------------------------------------
    def alternar_menu(self) -> None:
        self.aberto = not self.aberto
        self.destino = 1.0 if self.aberto else 0.0
        if self.aberto:
            self._mostrar_painel(True)
        self.atualizar_cores()
        self._iniciar_animacao()
        self.janela.Render()

    def fechar(self) -> None:
        if self.aberto:
            self.alternar_menu()

    def alternar_parte(self, posicao: int) -> None:
        parte = self.partes[posicao]
        parte["visivel"] = not parte["visivel"]
        parte["ator"].SetVisibility(parte["visivel"])
        self.atualizar_cores()
        self.janela.Render()

    def alternar_regua(self) -> None:
        self.regua.alternar()
        self.atualizar_cores()
        self.janela.Render()

    # -- ligacao com o mouse -------------------------------------------------
    def arrastando(self) -> bool:
        """True enquanto a regua estiver sendo puxada."""
        return self.regua.arrastando

    def arrastar(self, x: float, y: float) -> None:
        self.regua.arrastar(x, y)

    def soltar(self) -> None:
        self.regua.soltar()

    def clique(self, x: float, y: float) -> bool:
        """Trata um clique. Devolve True se o menu consumiu o clique."""
        if self.botao.contem(x, y):
            self.alternar_menu()
            return True

        if self.botao_slider.contem(x, y):
            self.alternar_regua()
            return True

        if self.aberto:
            for posicao, item in enumerate(self.itens):
                if item["caixa"].contem(x, y):
                    self.alternar_parte(posicao)
                    return True
            if self.painel.contem(x, y):
                return True  # clique no vazio do painel: so ignora

        if self.regua.pegar(x, y):
            return True

        if self.aberto:
            self.fechar()
            return True
        return False
