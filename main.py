import sys
sys.path.append("../gSpan-master")
from gspan_mining.config import parser
from gspan_mining.main import main


args_str = '-s 5 -d False -l 3 -p True -w True mygraph.data'
FLAGS, _ = parser.parse_known_args(args=args_str.split())
gs = main(FLAGS)