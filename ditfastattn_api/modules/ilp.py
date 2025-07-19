from pyomo.environ import *
import torch

def form_head_method_name(head_num, method):
    method_abbr = ""
    if method == "output_share":
        method_abbr = "ast"
        window_size = "0"
    elif method == "cfg_share":
        method_abbr = "asc"
        window_size = "-1"
    elif "arrow_attn" or "window_attn" in method:
        if "cfg" in method:
            method_abbr = "ascaa"
            window_size = "-" + method.split("_")[-1]
        else:
            if "reorder" in method:
                method_abbr = "aar"
                window_size = "re" + method.split("_")[-1]
            else:
                method_abbr = "aa"
                window_size = method.split("_")[-1]
    return "_".join([method_abbr, window_size, str(head_num)])

def set_constraint(model, head_set, head_method_dict, alpha):
    # set constraint 
    for head in head_set:
        temp_name_list = []
        for name in head_method_dict.keys():
            if head_method_dict[name]['head'] == head:
                temp_name_list.append(name)
        # print(temp_name_list)
        constraint_name = "_".join(["con", str(head)])
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
        for name in head_method_dict.keys():
            lhs.append(getattr(model, name) * head_method_dict[name]['influence'])
        return sum(lhs) <= alpha
    model.con_alpha = Constraint(rule=ConsAlphaRule)

def set_objective(model, head_method_dict):
    def ObjRule(model):
        res = 0
        for name in head_method_dict.keys():
            res += head_method_dict[name]['latency'] * getattr(model,name)
        return res
    model.obj = Objective(rule=ObjRule, sense=maximize)

def solve_ip(influence_dict, latency_dict, alpha):
    res = []
    if alpha > 0:
        head_method_dict = {} 
        model = ConcreteModel()
        head_set = set()
        for i in influence_dict:
            head_num = i[0]
            method = i[1]
            influence = influence_dict[i]
            name = form_head_method_name(head_num,  method)
            if name not in head_method_dict:
                head_method_dict[name] = {}

            head_method_dict[name]['influence'] = influence
            head_method_dict[name]['latency'] = latency_dict[method]
            head_method_dict[name]['var'] = Var(domain=NonNegativeIntegers, bounds=(0,1))
            head_method_dict[name]['head'] = head_num
            head_method_dict[name]['method'] = method
            head_method_dict[name]['full_name'] = i[0]

            head_set.add(head_num)
            
            setattr(model, name, head_method_dict[name]['var'])
        # set constraint 
        set_constraint(model, head_set, head_method_dict, alpha)

        set_objective(model, head_method_dict)

        solver = SolverFactory('glpk')
        result = solver.solve(model)

        for name in head_method_dict.keys():
            if head_method_dict[name]['var']() == 1:
                res.append((head_method_dict[name]['full_name'], name.split("_")[1]))
    return res


    
