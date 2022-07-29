from pathlib import Path

# save as latex table
def savetable(table, file_name):
    fig_dir = Path("../reports/figures/")
    txt_file = open(fig_dir / file_name, 'w')
    string = table.to_latex()
    txt_file.write(string)
    txt_file.close()