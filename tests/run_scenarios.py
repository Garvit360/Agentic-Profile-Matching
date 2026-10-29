from matching_agent import AgentState, build_graph, invoke_with_node_trace

SCENARIOS = [
    "Find me candidates with React and 3+ years experience.",
    "Compare top 3.",
    "Why did John rank higher than Jane?",
    "Generate interview questions for the top candidate.",
    "Actually require 5+ years and AWS.",
]


def main() -> None:
    graph = build_graph()
    state = AgentState()
    for msg in SCENARIOS:
        print("\nYou:", msg)
        state = invoke_with_node_trace(graph, state, msg)
        print("\nAgent:\n", state.last_response)


if __name__ == "__main__":
    main()
