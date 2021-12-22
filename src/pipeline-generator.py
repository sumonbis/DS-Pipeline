import os
import ast
import pandas as pd
import astpretty
# from graphviz import Digraph

dict_file = "stages.csv"
dc = pd.read_csv(dict_file, index_col=0, squeeze=True, header=None).to_dict()

def main():
    rootdir = "../notebooks"
    w_file = "pipe.txt"
    p = ""
    Utils.flush(w_file)

    total = 0
    count_chk = 0
    count_prev = 0
    count_next = 0

    for subdir, dirs, files in os.walk(rootdir):
        project = subdir[len(rootdir)+1:].split('/')[0]
        if "/" not in project and len(project) > 0:
            if project != p:
                print("\n\nProject: " + project)
                Utils.write(w_file, "\nProject: " + project)
                p = project

        for file in files:

            filepath = subdir + os.sep + file

            if filepath.endswith(".py"):

                #print(filepath + "\n")
                Utils.write(w_file, "\n" + file)
                Utils.write(w_file, "\n")
                try:
                    tree = getast(filepath)
                    # astpretty.pprint(tree, show_offsets=False)
                except Exception as e:
                    print("Error parsing AST: " + str(e))
                    continue

                visitor = FuncLister()
                visitor.s_list = []
                visitor.f_name = []
                visitor.f_dict = {}
                visitor.symb_dict = {}
                visitor.arg_arr = []
                visitor.symb_arr = []
                visitor.visit(tree)

                # for s in visitor.s_list:
                #     write_file(s, visitor.f_dict)
                edges = []
                pipe = []

                # print(visitor.s_list)
                for i in range(len(visitor.s_list)):
                    ins = list(set(visitor.arg_arr[i]))
                    outs = list(set(visitor.symb_arr[i]))
                    for j in ins:
                        found = False
                        for k in range(i - 1, -1, -1):
                            for sym in visitor.symb_arr[k]:
                                if (j == sym):
                                    edges.append((k, i))
                                    found = True
                                    break
                            if (found):
                                break

                    rec = []
                    build_pipe(visitor.s_list[i], visitor.f_dict, pipe, rec)

                print(pipe)

                chk = False

                total += len(pipe)

                for i in range(len(pipe)):
                    s = pipe[i]
                    Utils.write(FuncLister.write_path, get_acronym(s) + " -> ")
                    # print(total)



    print('\n' + str(total) + ' prev: ' + str(count_prev) + ', next: ' + str(count_next))

def build_pipe(s, dict, pipe, rec):
    if (s.startswith("9")):
        s = s[1:]
        if(len(dict[s]) > 0):
            for ss in dict[s]:
                if len(rec) < 100:
                    rec.append(ss)
                    build_pipe(ss, dict, pipe, rec)
            rec = []
    else:
        if (len(pipe) == 0):
            pipe.append(s)
        else:
            if (pipe[-1] != s):
                pipe.append(s)



def get_acronym(s):
    if (s == "1"):
        return "A"
    elif (s == "2"):
        return "Pr"
    elif (s == "3"):
        return "M"
    elif (s == "4"):
        return "Tr"
    elif (s == "5"):
        return "E"
    elif(s == "7"):
        return "Pd"
    else:
        return s

def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def getast(path):
    with open(path, "r") as source:
        tree = ast.parse(source.read())
        return tree


class FuncLister(ast.NodeVisitor):
    write_path = "pipe.txt"
    trailler = ""
    isClass = False
    isFunc = 0

    symb_dict = {}
    s_list = []
    f_name = []
    f_dict = {}
    arg_arr = []
    symb_arr = []
    symb = []

    def visit_ClassDef(self, node):
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self.isFunc += 1
        self.f_name.append(node.name)
        self.f_dict[self.f_name[-1]] = []
        self.generic_visit(node)
        self.f_name.pop()
        self.isFunc -= 1

    def visit_Assign(self, node):
        self.symb = []
        sym = ""
        for target in node.targets:
            if(isinstance(target, ast.Tuple)):
                for e in target.elts:
                    sym = self.getSymbol(e)
                    self.symb.append(sym)
                    self.symb_dict[sym] = "-"
            else:
                sym = self.getSymbol(target)
                self.symb.append(sym)
                self.symb_dict[sym] = "-"
        self.generic_visit(node)
        self.symb = []

    def getSymbol(self, target):
        if (isinstance(target, ast.Name)):
            return target.id
        elif (isinstance(target, ast.Subscript)):
            if (isinstance(target.value, ast.Name) and isinstance(target.slice, ast.Index) and isinstance(
                    target.slice.value, ast.Name)):
                return target.value.id + "[" + target.slice.value.id + "]"
        elif(isinstance(target, ast.Attribute)):
            return target.attr
        else:
            return "?"


    def visit_Call(self, node):
        api = ""
        ar = ""
        arg = []
        if isinstance(node.func, ast.Name):
            # print(ast.dump(node))
            try:
                arg = FuncLister.get_args(node)
                ar = str(arg)
            except:
                ar = "[?]"
            api_name = str(node.func.id)
            if(api_name.endswith('app.run')):
                api_name = 'main'
            api = api_name + " " + ar

        elif isinstance(node.func, ast.Attribute):
            #print(ast.dump(node))
            n = node.func
            FuncLister.trailler = ""
            AttrLister().visit(n)
            tr = FuncLister.trailler
            if(tr.endswith('app.run')):
                tr = 'main'
            try:
                arg = FuncLister.get_args(node)
                ar = str(arg)
            except:
                ar = "[?]"
            api = tr + " " + ar

        if api != "":
            # print(api)
            s = Utils.get_stage(api, self.f_dict.keys())
            # print(api + ', ' + s)

            if (s == "0" or s == "8"):
                pass
            else:
                if (self.isFunc > 0):

                    if (len(self.f_dict[self.f_name[-1]]) == 0):
                        self.f_dict[self.f_name[-1]].append(s)
                    else:
                        if (self.f_dict[self.f_name[-1]][-1] != s):
                            self.f_dict[self.f_name[-1]].append(s)

                else:
                    if (len(self.s_list) == 0):
                        self.s_list.append(s)
                        self.arg_arr.append(arg)
                        self.symb_arr.append(self.symb)

                    else:
                        if (self.s_list[-1] != s):
                            self.s_list.append(s)
                            self.arg_arr.append(arg)
                            self.symb_arr.append(self.symb)
                        else:
                            self.arg_arr[-1].extend(arg)
                            self.symb_arr[-1].extend(self.symb)

        self.generic_visit(node)


    def get_args(node):
        a = []
        b = []
        for arg in node.args:
            if isinstance(arg, ast.Starred):
                a.append(arg.value.id)
            #elif isinstance(arg, ast.Str):
            elif isinstance(arg, ast.BinOp):
                a.append(Utils().get_val(arg.left) + Utils().get_bin_op(arg.op) + Utils().get_val(arg.right))
            else:
                a.append(Utils().get_val(arg))
                if (isinstance(arg, ast.Name)):
                    b.append(Utils().get_val(arg)) ###
        for kw in node.keywords:
            if isinstance(kw, ast.keyword):
                if kw.arg == None:
                    a.append("**" + kw.value.id)
                else:
                    a.append(kw.arg + "=" + Utils().get_val(kw.value))
        return b


class AttrLister(ast.NodeVisitor):
    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Attribute):
            if FuncLister.trailler == "":
                FuncLister.trailler = node.attr
            else:
                FuncLister.trailler = node.attr + "." + FuncLister.trailler
        if isinstance(node.value, ast.Name):
            if FuncLister.trailler == "":
                FuncLister.trailler = node.value.id + "." + node.attr
            else:
                FuncLister.trailler = node.value.id + "." + node.attr + "." + FuncLister.trailler
        self.generic_visit(node)


class Utils:
    def get_val(self, node):
        if isinstance(node, ast.Num):
            return str(node.n)
        elif isinstance(node, ast.Str):
            return str(node.s)
        elif isinstance(node, ast.Name):
            return str(node.id)
        elif isinstance(node, ast.NameConstant):
            return str(node.value)
        elif isinstance(node, ast.Call):
            return "CALL"
        elif isinstance(node, ast.Subscript):
            return str(Utils().get_val(node.value)) # + handle subcript Slice(Index, Slice or ExtSlice)
        elif isinstance(node, ast.Attribute):
            FuncLister.trailler = ""
            AttrLister().visit(node)
            return FuncLister.trailler
        elif isinstance(node, ast.List):
            return str(Utils().get_elts(node))
        elif isinstance(node, ast.Tuple):
            return str(Utils().get_elts(node))
        else:
            return "UNKNOWN"

    def get_elts(self, node):
        a = []
        for e in node.elts:
            a.append(Utils().get_val(e))
        return str(a)

    def get_bin_op(self, node):
        if isinstance(node, ast.Add):
            return " + "
        elif isinstance(node, ast.Sub):
            return " - "
        elif isinstance(node, ast.Mult):
            return " * "
        elif isinstance(node, ast.Div):
            return " / "
        elif isinstance(node, ast.FloorDiv):
            return " // "
        elif isinstance(node, ast.Mod):
            return " % "
        elif isinstance(node, ast.Pow):
            return " ** "
        elif isinstance(node, ast.LShift):
            return " << "
        elif isinstance(node, ast.RShift):
            return " >> "
        elif isinstance(node, ast.BitAnd):
            return " B_AND "
        elif isinstance(node, ast.BitOr):
            return " B_OR "
        elif isinstance(node, ast.BitXor):
            return " B_XOR "

    def write(path, content):
        with open(path, 'a') as file:
            file.write(content)

    def flush(path):
        open(path, 'w').close()

    def get_stage(api, fs):
        name = api.split(" [")[0]
        parts = name.split(".")
        root = parts[-1]

        if root in fs:
            s = "9" + root
            return s

        for a in dc:
            if name.endswith(a):
                st = str(dc.get(a))
                if(st == 'None'):
                    print(a)
                return st
        return "0"


if __name__ == "__main__":
    main()
