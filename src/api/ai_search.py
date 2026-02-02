"""
AI-powered search for Portal IQ using Claude.

Allows natural language queries over the player database:
- "Show me 4-star QBs in the portal with 3000+ passing yards"
- "Find undervalued players outperforming their recruiting ranking"
- "Top NIL prospects from SEC schools"
"""

import os
import json
import logging
from typing import Optional, List, Dict, Any
import anthropic
import pandas as pd
from pathlib import Path

logger = logging.getLogger("portal_iq_api")

# Data paths
DATA_DIR = Path("data/raw")

class AISearch:
    """AI-powered search using Claude to query player data."""

    def __init__(self):
        self.client = None
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if self.api_key:
            self.client = anthropic.Anthropic(api_key=self.api_key)
            logger.info("AISearch initialized with Anthropic API")
        else:
            logger.warning("ANTHROPIC_API_KEY not set - AI search disabled")

        # Load data into memory for querying
        self.data = self._load_data()

    def _load_data(self) -> Dict[str, pd.DataFrame]:
        """Load available datasets."""
        data = {}

        files = {
            "players": "player_season_stats.csv",
            "rosters": "team_rosters.csv",
            "recruiting": "recruiting_rankings.csv",
            "portal": "transfer_portal.csv",
            "talent": "team_talent.csv",
            "draft": "nfl_draft_picks.csv"
        }

        for name, filename in files.items():
            path = DATA_DIR / filename
            if path.exists():
                try:
                    data[name] = pd.read_csv(path)
                    logger.info(f"Loaded {name}: {len(data[name])} records")
                except Exception as e:
                    logger.warning(f"Failed to load {filename}: {e}")

        return data

    def _get_schema_summary(self) -> str:
        """Generate a summary of available data for Claude."""
        summaries = []

        for name, df in self.data.items():
            cols = df.columns.tolist()[:15]  # First 15 columns
            sample = df.head(2).to_dict(orient='records') if len(df) > 0 else []
            summaries.append(f"""
**{name}** ({len(df)} records)
Columns: {', '.join(cols)}
Sample: {json.dumps(sample, default=str)[:500]}
""")

        return "\n".join(summaries)

    def is_available(self) -> bool:
        """Check if AI search is available."""
        return self.client is not None and len(self.data) > 0

    async def search(
        self,
        query: str,
        max_results: int = 25
    ) -> Dict[str, Any]:
        """
        Execute a natural language search query.

        Args:
            query: Natural language query like "top portal QBs from SEC"
            max_results: Maximum results to return

        Returns:
            Dict with results, explanation, and query interpretation
        """
        if not self.client:
            return {
                "error": "AI search not available - ANTHROPIC_API_KEY not configured",
                "results": []
            }

        if not self.data:
            return {
                "error": "No data loaded - run data collection first",
                "results": []
            }

        # Build the prompt for Claude
        schema = self._get_schema_summary()

        system_prompt = f"""You are a college football data analyst assistant for Portal IQ.
You help users query player data using natural language.

Available datasets:
{schema}

When given a query, you must:
1. Understand what the user is looking for
2. Write Python pandas code to filter/query the data
3. Return ONLY valid Python code that will execute

The code should:
- Use the `data` dict which contains DataFrames: {list(self.data.keys())}
- Assign the final result to a variable called `results`
- Limit results to {max_results} rows
- Handle missing columns gracefully
- Return a DataFrame

Example for "4-star QBs in portal":
```python
if 'portal' in data and 'recruiting' in data:
    portal_df = data['portal']
    recruiting_df = data['recruiting']
    # Merge to get star ratings
    merged = portal_df.merge(recruiting_df, left_on='name', right_on='name', how='left')
    results = merged[merged['stars'] >= 4].head({max_results})
else:
    results = pd.DataFrame()
```

Return ONLY the Python code block, no explanations."""

        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": f"Query: {query}"}
                ]
            )

            # Extract code from response
            response_text = message.content[0].text
            code = self._extract_code(response_text)

            if not code:
                return {
                    "error": "Could not generate query code",
                    "query": query,
                    "results": []
                }

            # Execute the code safely
            results_df = self._execute_query(code)

            if results_df is None or len(results_df) == 0:
                return {
                    "query": query,
                    "interpretation": "Query executed but returned no results",
                    "code": code,
                    "results": [],
                    "count": 0
                }

            # Convert results to list of dicts
            results_list = results_df.head(max_results).to_dict(orient='records')

            # Clean up NaN values
            for row in results_list:
                for key, value in row.items():
                    if pd.isna(value):
                        row[key] = None

            return {
                "query": query,
                "interpretation": f"Found {len(results_df)} matching records",
                "results": results_list,
                "count": len(results_df),
                "code": code
            }

        except Exception as e:
            logger.error(f"AI search error: {e}")
            return {
                "error": str(e),
                "query": query,
                "results": []
            }

    def _extract_code(self, response: str) -> Optional[str]:
        """Extract Python code from Claude's response."""
        # Look for code blocks
        if "```python" in response:
            start = response.find("```python") + 9
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()

        if "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()

        # No code block, try to use whole response
        return response.strip()

    def _execute_query(self, code: str) -> Optional[pd.DataFrame]:
        """Safely execute the generated query code."""
        try:
            # Create a safe execution environment
            local_vars = {
                "data": self.data,
                "pd": pd,
                "results": pd.DataFrame()
            }

            # Execute the code
            exec(code, {"__builtins__": {}}, local_vars)

            results = local_vars.get("results")

            if isinstance(results, pd.DataFrame):
                return results
            elif isinstance(results, pd.Series):
                return results.to_frame()
            else:
                return None

        except Exception as e:
            logger.error(f"Query execution error: {e}")
            logger.error(f"Code was: {code}")
            return None


# Singleton instance
_ai_search: Optional[AISearch] = None

def get_ai_search() -> AISearch:
    """Get or create the AI search instance."""
    global _ai_search
    if _ai_search is None:
        _ai_search = AISearch()
    return _ai_search
