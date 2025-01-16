from pyomo.environ import *
import torch

def form_layer_method_name(layer_num, layer_type, method):
    method_abbr = ""
    if method == "output_share":
        method_abbr = "ast"
    elif "window_attn" in method:
        method_abbr = "wars"
    return "_".join([layer_type, layer_num, method_abbr])

def set_constraint(model, layer_set, layer_method_dict, alpha):
    # set constraint 
    for layer in layer_set:
        temp_name_list = []
        for name in layer_method_dict.keys():
            if layer_method_dict[name]['layer'] == layer:
                temp_name_list.append(name)
        # print(temp_name_list)
        constraint_name = "_".join(["con", layer])
        if len(temp_name_list) > 1:
            def ConsRule(model):
                lhs = []
                for i in temp_name_list:
                    # print(i)
                    lhs.append(getattr(model, i))
                return sum(lhs) <= 1
            temp_con = Constraint(rule=ConsRule)
            # print(constraint_name)
            setattr(model, constraint_name, temp_con)

    # alpha = 0.0001
    def ConsAlphaRule(model):
        lhs = []
        for name in layer_method_dict.keys():
            lhs.append(getattr(model, name) * layer_method_dict[name]['influence'])
        return sum(lhs) <= alpha
    model.con_alpha = Constraint(rule=ConsAlphaRule)

def set_objective(model, layer_method_dict):
    def ObjRule(model):
        res = 0
        for name in layer_method_dict.keys():
            res += layer_method_dict[name]['latency'] * getattr(model,name)
        return res
    model.obj = Objective(rule=ObjRule, sense=maximize)

def solve_ip(influence_dict, latency_dict, alpha):
    res = []
    if alpha > 0:
        layer_method_dict = {} 
        model = ConcreteModel()
        layer_set = set()
        for i in influence_dict:
            _, layer_num, layer_type = i[0].split(".")
            method = i[1]
            influence = i[2].item()
            name = form_layer_method_name(layer_num, layer_type, method)
            if name not in layer_method_dict:
                layer_method_dict[name] = {}

            current_layer = "_".join([layer_type, layer_num])
            layer_method_dict[name]['influence'] = influence
            layer_method_dict[name]['latency'] = latency_dict[(layer_type, method)]
            layer_method_dict[name]['var'] = Var(domain=NonNegativeIntegers, bounds=(0,1))
            layer_method_dict[name]['layer'] = current_layer
            layer_method_dict[name]['method'] = method
            layer_method_dict[name]['full_name'] = i[0]

            layer_set.add(current_layer)
            
            setattr(model, name, layer_method_dict[name]['var'])

        # set constraint 
        set_constraint(model, layer_set, layer_method_dict, alpha)

        set_objective(model, layer_method_dict)

        # breakpoint()
        # print("start solving ...")

        # 求解器
        solver = SolverFactory('glpk')  # 使用GLPK求解器
        result = solver.solve(model)

        # 输出结果
        print("Status:", result.solver.status)
        print(result['Solver'][0]['Termination condition'])
        for name in layer_method_dict.keys():
            print(f"{name}: {layer_method_dict[name]['var']()}")
            if layer_method_dict[name]['var']() == 1:
                res.append((layer_method_dict[name]['full_name'], layer_method_dict[name]['method']))
        breakpoint()
    return res


    
