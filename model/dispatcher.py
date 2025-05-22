from model.models import *

class Dispatcher(object):
    
    def __init__(self):
        pass

    def dispatch(self, node_model_type, args):
        """
            Dispatch the model based on the node_model_type.
            Input:
                node_model_type: str, type of the model
                args: arguments for the model
        """

        if node_model_type == "GCN":
            model = GCN(in_channels=args.in_dim, hidden_channels=args.hidden_dim, out_channels=args.out_dim)
        elif node_model_type == "GAT":
            model = GAT(in_channels=args.in_dim, hidden_channels=args.hidden_dim, out_channels=args.out_dim)
        elif node_model_type == "SAGE":
            model = SAGE(in_channels=args.in_dim, hidden_channels=args.hidden_dim, out_channels=args.out_dim)
        elif node_model_type == "MLP":
            model = MLP(in_channels=args.in_dim, hidden_channels=args.hidden_dim, out_channels=args.out_dim)
        elif node_model_type == "SGC":
            model = SGC(in_channels=args.in_dim, out_channels=args.out_dim, K=2)
        
        return model