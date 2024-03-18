from elisa_compile import ELISACOMPILE

if __name__ == '__main__':
    master_fl = "/Users/gorkalasso/Documents/GitHub/filo_GP-bat_NPC1/scr/elisa/input_master.txt"
    elisa_comp = ELISACOMPILE(master_fl)
    elisa_comp.read_infl()