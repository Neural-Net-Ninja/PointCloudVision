class TestDropout(unittest.TestCase):
    """
    Temporarily tests the dropout layer of KP-FCNN.  # ToDo: Remove this test once the dropout layer is tested properly.
    # TODO: 2 things need to be figured out:
            # 1. If dropout adds value to kpfcnn
            # 2. If it does, what should be the default dropout value
    """
    def setUp(self) -> None:
        set_deterministic_mode('full')
        self.input_dim = 3
        self.num_classes = 3

        self.cuda_available = torch.cuda.is_available()
        self.torch_device = torch.device("cuda:0" if self.cuda_available else "cpu")
    
    def test_dropout(self):
        
        # Create some random input data and labels
        batch_size = 2
        point_cloud_size = 128
        training_coords = torch.randn(batch_size, point_cloud_size, 3).to(self.torch_device)
        labels = torch.randint(0, self.num_classes - 1, [batch_size*point_cloud_size]).long().to(self.torch_device)
        
        validation_losses = {}
        dropout_values = [drop / 10.0 for drop in range(10)] # Range of dropout values to test
        for dropout in dropout_values:
            # Create an instance of the network
            kpfcnn = Kpfcnn(self.num_classes, self.input_dim, dropout, input_type='variable').to(self.torch_device)
            #kpfcnn.reset_parameters()

            criterion = torch.nn.CrossEntropyLoss().to(self.torch_device)
            optimizer = kpfcnn.configure_optimizer(0.0001, 0.0)

            # Compute the loss with dropout turned on
            kpfcnn.train()
            prediction_with_dropout = kpfcnn(training_coords)
            prediction_with_dropout = prediction_with_dropout.reshape(-1, self.num_classes)
            print('prediction_with_dropout:____',prediction_with_dropout.shape)
            print('labels:____',labels.shape)
            loss_with_dropout = kpfcnn.loss(criterion, prediction_with_dropout, labels)

            # Compute the loss with dropout turned off
            kpfcnn.eval()
            prediction_no_dropout = kpfcnn(training_coords)
            loss_no_dropout = criterion(prediction_no_dropout, labels)
            
            # When experimenting with different values of dropout, one way to determine which value works
            # best is by looking at the validation loss.
            validation_losses[dropout] = (loss_with_dropout.item(), loss_no_dropout.item())

        # Find the dropout value that gives the lowest validation loss
        print('validation_losses:____',validation_losses)
        losses_with_no_dropout = [loss[1] for loss in validation_losses.values()]
        loss_no_dropout = min(losses_with_no_dropout)
        for dropout, loss in validation_losses.items():
            if loss[1] == loss_no_dropout:
                loss_with_dropout = loss[0]
                print('Dropout value that gives the lowest validation loss:____', dropout)
                break

        # Ensure that the loss with dropout is greater than the loss without dropout
        # check for the case were validation loss is minimum of all losses with without dropout
        self.assertGreater(loss_with_dropout, loss_no_dropout, "Dropout is not working properly")