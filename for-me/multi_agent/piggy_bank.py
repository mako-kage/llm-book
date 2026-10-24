from typing import Annotated, TypedDict 
import operator 
from langgraph.graph import StateGraph, END
import functools 

class PiggyBankState(TypedDict):
  total: Annotated[int, operator.add]
  count: Annotated[int, operator.add]
  last_deposit: int

def deposit(state: PiggyBankState) -> dict: 
  amount = int(input("Enter the amount to deposit: "))
  return {"total": amount, "count": 1, "last_deposit": amount}

def finalize(state: PiggyBankState) -> dict: 
  print(f"{state['count']}回の貯金で目標金額に到達しました。")
  print(f"{state['total']}円貯まっています。")
  print(f"最後の入金額は{state['last_deposit']}円でした。")
  return {"total": 0}

def check_goal(state: PiggyBankState, goal: int) -> dict: 
  if state["total"] >= goal:
    return "full"
  else: 
    return "continue"

def piggy_bank(goal: int):
  workflow = StateGraph(PiggyBankState)
  workflow.add_node("Deposit", deposit)
  workflow.add_node("Full", finalize)
  workflow.add_conditional_edges(
    "Deposit", 
    functools.partial(check_goal, goal=goal), 
    {"continue": "Deposit", "full": "Full"}, 
  )
  workflow.add_edge("Full", END)
  workflow.set_entry_point("Deposit")
  graph = workflow.compile()
  final_state = graph.invoke({"total": 0, "count": 0, "last_deposit": 0})
  print(final_state)

if __name__ == "__main__":
  piggy_bank(1000)