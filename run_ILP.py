import pulp


def solve_ilp_from_json(path: str, msg: bool = True):
    print('Loading model from JSON:', path)
    variables, model = pulp.LpProblem.fromJson(path)
    print('Model loaded.')
    print('Solving...')
    status = model.solve(pulp.PULP_CBC_CMD(msg=msg))

    print("Status:", pulp.LpStatus[status])
    print("Objective:", pulp.value(model.objective))

    return model, variables, pulp.value(model.objective)

if __name__ == "__main__":
    path = r'C:\Oren\Academy\Thesis\Thesis-percolation\full_model_n6.json'
    solve_ilp_from_json(path)