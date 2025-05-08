from model.models import *

class Dispatcher(object):
    
    def __init__():
        pass

    def dispatch(self, args):
        if args.model_name == "GCN":
            model = GCN(in_channels=args.in_dim, hidden_channels=args.hidden_dim, out_channels=args.out_dim)
        elif args.model_name == "GAT":
            model = GAT(in_channels=args.in_dim, hidden_channels=args.hidden_dim, out_channels=args.out_dim)
        elif args.model_name == "SAGE":
            model = SAGE(in_channels=args.in_dim, hidden_channels=args.hidden_dim, out_channels=args.out_dim)
        elif args.model_name == "MLP":
            model = MLP(in_channels=args.in_dim, hidden_channels=args.hidden_dim, out_channels=args.out_dim)
        elif args.model_name == "SGC":
            model = SGC(in_channels=args.in_dim, out_channels=args.out_dim, K=2)
        
        return model