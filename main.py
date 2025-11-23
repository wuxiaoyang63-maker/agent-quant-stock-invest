# ============================================================================
# Main Entry Point
# ============================================================================

import asyncio

from dotenv import load_dotenv
from agent_workflow import QuantInvestWorkflow

async def main() -> None:
    """
    Main execution function demonstrating the correct Agent Framework pattern.

    Demonstrates:
    - Loading environment variables
    - Creating workflow orchestrator
    - Running analysis task
    - Printing results
    """
    # Load environment variables
    load_dotenv()

    # Initialize workflow orchestrator
    workflow = QuantInvestWorkflow()

    # Create the workflow (agents + executors + edges)
    built_workflow = await workflow.create_workflow()

    # Define analysis task
    task = """
    趋势上涨强势阶段(连续两日上涨，且两日累计涨幅超过10%，两日每天持续放量10%，或者是对比30日平均成交额明显放量且前一天收盘价在5日线上)
1，强势阶段竞价低开:低开0-1个点，且无明显大资金流出，属于买入点
2，强势阶段竞价低开:低开1-2个点，无明显大资金流出，开盘3分钟内在-3附近买入
3强势阶段竞价平开:竞价对比前两天无明显缩量(5%)以内，前日涨幅不超过5%，前两日涨幅不超过15%，买入。
    """

    # Run the workflow
    await workflow.run_task(built_workflow, task)

    # This server needs to be outside the workflow run to keep it alive
    # Launch server with the workflow
    # from agent_framework.devui import serve
    # Expose the workflow at http://localhost:8090
    # serve(entities=[built_workflow], port=8090, auto_open=True)

    print("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
