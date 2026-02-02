"""
Visualization Utilities

Chart generation for dashboards and reports.
"""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Optional imports for visualization
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class ChartGenerator:
    """Generate charts for ML Engine dashboards."""

    # Color schemes
    COLORS = {
        "primary": "#1f77b4",
        "secondary": "#ff7f0e",
        "success": "#2ca02c",
        "danger": "#d62728",
        "warning": "#ffbb33",
        "info": "#17becf",
    }

    TEAM_COLORS = {
        "ARI": "#97233F", "ATL": "#A71930", "BAL": "#241773", "BUF": "#00338D",
        "CAR": "#0085CA", "CHI": "#C83803", "CIN": "#FB4F14", "CLE": "#311D00",
        "DAL": "#003594", "DEN": "#FB4F14", "DET": "#0076B6", "GB": "#203731",
        "HOU": "#03202F", "IND": "#002C5F", "JAX": "#006778", "KC": "#E31837",
        "LAC": "#0080C6", "LAR": "#003594", "LV": "#000000", "MIA": "#008E97",
        "MIN": "#4F2683", "NE": "#002244", "NO": "#D3BC8D", "NYG": "#0B2265",
        "NYJ": "#125740", "PHI": "#004C54", "PIT": "#FFB612", "SEA": "#002244",
        "SF": "#AA0000", "TB": "#D50A0A", "TEN": "#0C2340", "WAS": "#773141",
    }

    def __init__(self):
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly required for visualization")

    def create_bar_chart(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        title: str,
        color: Optional[str] = None,
        horizontal: bool = False,
    ) -> go.Figure:
        """Create a bar chart."""
        if horizontal:
            fig = px.bar(df, x=y, y=x, title=title, color=color, orientation="h")
        else:
            fig = px.bar(df, x=x, y=y, title=title, color=color)

        fig.update_layout(
            template="plotly_white",
            title_x=0.5,
        )
        return fig

    def create_line_chart(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        title: str,
        color: Optional[str] = None,
    ) -> go.Figure:
        """Create a line chart."""
        fig = px.line(df, x=x, y=y, title=title, color=color)
        fig.update_layout(
            template="plotly_white",
            title_x=0.5,
        )
        return fig

    def create_scatter_chart(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        title: str,
        color: Optional[str] = None,
        size: Optional[str] = None,
        hover_data: Optional[List[str]] = None,
    ) -> go.Figure:
        """Create a scatter plot."""
        fig = px.scatter(
            df, x=x, y=y, title=title,
            color=color, size=size, hover_data=hover_data,
        )
        fig.update_layout(
            template="plotly_white",
            title_x=0.5,
        )
        return fig

    def create_pie_chart(
        self,
        df: pd.DataFrame,
        values: str,
        names: str,
        title: str,
    ) -> go.Figure:
        """Create a pie chart."""
        fig = px.pie(df, values=values, names=names, title=title)
        fig.update_layout(
            template="plotly_white",
            title_x=0.5,
        )
        return fig

    def create_heatmap(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        z: str,
        title: str,
    ) -> go.Figure:
        """Create a heatmap."""
        pivot = df.pivot(index=y, columns=x, values=z)
        fig = px.imshow(pivot, title=title, aspect="auto")
        fig.update_layout(
            template="plotly_white",
            title_x=0.5,
        )
        return fig

    def create_cap_breakdown_chart(
        self,
        team_data: Dict[str, float],
        title: str = "Cap Breakdown",
    ) -> go.Figure:
        """Create salary cap breakdown chart."""
        labels = list(team_data.keys())
        values = list(team_data.values())

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            textinfo="label+percent",
        )])

        fig.update_layout(
            title=title,
            title_x=0.5,
            template="plotly_white",
        )
        return fig

    def create_player_comparison_radar(
        self,
        players: List[Dict[str, Any]],
        metrics: List[str],
        title: str = "Player Comparison",
    ) -> go.Figure:
        """Create radar chart for player comparison."""
        fig = go.Figure()

        for player in players:
            values = [player.get(m, 0) for m in metrics]
            values.append(values[0])  # Close the polygon

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics + [metrics[0]],
                name=player.get("name", "Player"),
                fill="toself",
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title=title,
            title_x=0.5,
            template="plotly_white",
        )
        return fig

    def create_contract_timeline(
        self,
        contracts: List[Dict[str, Any]],
        title: str = "Contract Timeline",
    ) -> go.Figure:
        """Create contract timeline visualization."""
        fig = go.Figure()

        for i, contract in enumerate(contracts):
            fig.add_trace(go.Bar(
                x=[contract.get("years", 0)],
                y=[contract.get("player_name", f"Player {i}")],
                orientation="h",
                name=contract.get("player_name", ""),
                text=f"${contract.get('aav', 0) / 1_000_000:.1f}M AAV",
                textposition="inside",
            ))

        fig.update_layout(
            title=title,
            title_x=0.5,
            xaxis_title="Contract Years",
            template="plotly_white",
            showlegend=False,
        )
        return fig

    def create_nil_tier_distribution(
        self,
        df: pd.DataFrame,
        value_col: str = "nil_value",
        title: str = "NIL Value Distribution",
    ) -> go.Figure:
        """Create NIL tier distribution chart."""
        # Define tier bins
        bins = [0, 25_000, 100_000, 500_000, 1_000_000, float("inf")]
        labels = ["Entry", "Moderate", "Solid", "Premium", "Mega"]

        df = df.copy()
        df["tier"] = pd.cut(df[value_col], bins=bins, labels=labels)
        tier_counts = df["tier"].value_counts()

        fig = px.bar(
            x=tier_counts.index,
            y=tier_counts.values,
            title=title,
            labels={"x": "NIL Tier", "y": "Player Count"},
        )

        fig.update_layout(
            template="plotly_white",
            title_x=0.5,
        )
        return fig

    def create_surplus_value_chart(
        self,
        df: pd.DataFrame,
        player_col: str = "player_name",
        surplus_col: str = "surplus_value",
        title: str = "Surplus Value by Player",
        top_n: int = 20,
    ) -> go.Figure:
        """Create surplus value bar chart."""
        top_players = df.nlargest(top_n, surplus_col)

        colors = [
            self.COLORS["success"] if v > 0 else self.COLORS["danger"]
            for v in top_players[surplus_col]
        ]

        fig = go.Figure(data=[go.Bar(
            x=top_players[surplus_col] / 1_000_000,
            y=top_players[player_col],
            orientation="h",
            marker_color=colors,
        )])

        fig.update_layout(
            title=title,
            title_x=0.5,
            xaxis_title="Surplus Value ($M)",
            template="plotly_white",
            height=max(400, top_n * 25),
        )
        return fig

    def get_team_color(self, team: str) -> str:
        """Get team color code."""
        return self.TEAM_COLORS.get(team.upper(), self.COLORS["primary"])
