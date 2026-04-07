from __future__ import annotations
from typing import Any

import graphviz


class Module:
    def __init__(self, name, pos: tuple = (), **node_format) -> None:
        self.name = name
        self.pos = pos
        self.node_format = node_format
        self.out_id = self.name + " " + str(self.pos)

    def __call__(self, *args: Any, **kwds: Any) -> str:
        return self.f(*args, **kwds)

    def f(self, G: graphviz.Graph, *X: str) -> str:

        G.node(self.out_id, self.name, **self.node_format)

        for i in X:
            G.edge(i, self.out_id)
        return self.out_id


class FractalTransformerGraph(Module):
    def __init__(
        self,
        depth: int,
        pos: tuple = (),
    ) -> None:
        super().__init__(
            "T" + str(depth),
            pos,
            shape="diamond",
            style="filled",
            fillcolor="gray",
        )

        self.depth = depth

        self.plus = Module(
            "+",
            pos,
            shape="ellipse",
            fixedsize="true",
            width=".25",
            height=".25",
            style="filled",
            fillcolor="limegreen",
        )
        self.pre = self.recurse(depth - 1, pos + ("pre",))
        self.mid = FractalAttentionGraph(depth, pos + ("mid",))
        self.end = self.recurse(depth - 1, pos + ("end",))

    def recurse(self, depth, pos):
        if depth == 1:
            return Module(
                "MLP",
                pos,
                shape="diamond",
                style="filled",
                fillcolor="orchid",
            )
        else:
            return FractalTransformerGraph(depth, pos)

    def f(self, B: graphviz.Graph, *X) -> str:
        X = super().f(B, *X)
        with B.subgraph(
            name=self.out_id,
            comment=self.name,
            graph_attr={"cluster": "true", "pencolor": "limegreen"},
        ) as G:  # type: ignore
            X = self.pre(G, X)
            X_mid = self.mid(G, X)
            X = self.plus(G, X, X_mid)
            X = self.end(G, X)
        # X = super().f(B, X)
        return X


class FractalAttentionGraph(Module):
    steps = [
        "Q",
        "K",
        "VI",
        "A",
        "VO",
    ]

    def __init__(
        self,
        depth,
        pos: tuple = (),
        step="",
    ):
        super().__init__(
            "A" + str(depth) + " " + step,
            pos,
            shape="rect",
            style="filled",
            fillcolor="lightgray",
            # fillcolor="orange",
        )
        (
            self.to_query,
            self.to_key,
            self.to_value,
            self.to_attention_logits,
            self.to_value_out,
        ) = (
            (
                Module(
                    "L " + i,
                    pos + (i,),
                    shape="rect",
                    style="filled",
                    fillcolor="tomato",
                )
            )
            if depth == 2
            else (FractalAttentionGraph(depth - 1, pos + (i,), i))
            for i in self.steps
        )

        (
            self.plus_query,
            self.plus_key,
            self.plus_value,
            self.plus_attention_logits,
            self.plus_value_out,
        ) = (
            (
                Module(
                    "+",
                    pos + (i,),
                    shape="ellipse",
                    fixedsize="true",
                    width=".25",
                    height=".25",
                    style="filled",
                    fillcolor="limegreen",
                )
            )
            for i in self.steps
        )

        (
            self.key_times_query,
            self.value_times_attention_selection,
        ) = (
            (
                Module(
                    "*",
                    pos + (i,),
                    shape="diamond",
                    fixedsize="true",
                    width=".25",
                    height=".25",
                    style="filled",
                    fillcolor="orange",
                )
            )
            for i in ["KQ", "AV"]
        )

        self.select = Module(
            "softmax",
            pos,
            shape="ellipse",
            fixedsize="true",
            width=".75",
            height=".25",
            style="filled",
            fillcolor="orange",
        )
        self.sum = Module(
            "sum",
            pos,
            shape="ellipse",
            fixedsize="true",
            width=".5",
            height=".25",
            style="filled",
            fillcolor="orange",
        )

    def f(self, B, *X):
        X = super().f(B, *X)
        with B.subgraph(
            name=self.out_id,
            comment=self.name,
            graph_attr={"cluster": "true", "pencolor": "orange"},
        ) as G:  # type: ignore
            query = self.to_query(G, X)
            key = self.to_key(G, X)
            query = self.plus_query(G, X, query)
            key = self.plus_key(G, X, key)

            value = self.to_value(G, X)
            value = self.plus_value(G, X, value)

            attention_raw = self.key_times_query(G, key, query)

            attention_logits = self.plus_attention_logits(
                G, attention_raw, self.to_attention_logits(G, attention_raw)
            )

            attention_selection = self.select(G, attention_logits)

            value_selected = self.value_times_attention_selection(
                G, value, attention_selection
            )

            value_sum = self.sum(G, value_selected)

            value_out = self.plus_value_out(
                G, value_sum, self.to_value_out(G, value_sum)
            )

            X = value_out

        # X = super().f(B, X)
        return X


# meta_dims = 3


for meta_dims in range(2, 4 + 1):
    for graph in (graphviz.Digraph(format="png"), graphviz.Digraph(format="svg")):
        for model in (
            FractalTransformerGraph(meta_dims),
            FractalAttentionGraph(meta_dims),
        ):
            graph.node(
                "START",
                "START",
                shape="Mdiamond",
                style="filled",
                fillcolor="cornflowerblue",
            )
            graph.node(
                "END",
                "END",
                shape="Msquare",
                style="filled",
                fillcolor="cornflowerblue",
            )

            X = model(graph, "START")
            graph.edge(X, "END")

            graph.render(
                str(meta_dims) + "D " + type(model).__name__ + ".gv", "fractal_graphviz"
            )
            graph.clear()
