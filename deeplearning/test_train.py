import unittest
from deeplearning.train import *

class CloneRepoTest(unittest.TestCase):
    def test_model_init(self):
        model = TransformerBOWMixed(vocab_size=500)
        self.assertIsInstance(model, TransformerBOWMixed)

    def test_modular_one_pass(self):
        vocab_size = 500
        bs = 37

        model = TransformerBOWMixed(vocab_size=vocab_size)

        # some dummy
        for _ in range(100):
            input = np.random.uniform(size=vocab_size * bs).reshape((bs, vocab_size))
            input = torch.Tensor(input).float()
            target = torch.from_numpy(np.random.randint(0, 12, size=bs)).long()

            all_loss, lml, lat, lem, lvat, output = model.one_pass(input, target)
            self.assertGreater(lml,0)
            self.assertGreater(lat,0)
            self.assertGreater(lem,0)
            self.assertGreater(lvat,0)
            all_loss.backward()



    def test_train(self):
        import shutil

        def onerror(func, path, exc_info):
            """
            Error handler for ``shutil.rmtree``.

            If the error is due to an access error (read only file)
            it attempts to add write permission and then retries.

            If the error is for another reason it re-raises the error.

            Usage : ``shutil.rmtree(path, onerror=onerror)``
            """
            import stat
            if not os.access(path, os.W_OK):
                # Is the error an access error ?
                os.chmod(path, stat.S_IWUSR)
                func(path)
            else:
                raise PermissionError

        load=True
        vocab_size = 500
        max_len = 1000

        model = TransformerBOWMixed(vocab_size=vocab_size, d_model=128, d_inner=16, dropout=0.1, n_layers=4,
                                    xi=0.5)
        model.cuda()
        model.reset_parameters()

        optimizer = Adam(model.parameters(), lr=0.001)

        id_str = "test"
        task_dir = os.path.dirname(abspath(__file__))
        save_dir = Path(task_dir) / "saves" / id_str
        if save_dir.exists():
            shutil.rmtree(save_dir, onerror=onerror)


        if load:
            model, optimizer, highestepoch, highestiter = load_model(model, optimizer, 0, 0, id_str)
        else:
            highestepoch = 0

        logfile = dir_path / ("log/" + id_str)
        logfile.mkdir(exist_ok=True)
        logfile = logfile / (id_str + "_" + datetime_filename() + ".txt")

        # modify batch size here
        bs = 256

        tig = VocabIGBatchpkl(vocab_size=vocab_size, max_len=max_len, batch_size=bs, bow=True)
        vig = VocabIGBatchpkl(vocab_size=vocab_size, max_len=max_len, batch_size=bs, bow=True, valid=True)

        dq = deque(maxlen=100)
        mldq = deque(maxlen=100)
        atdq = deque(maxlen=100)
        emdq = deque(maxlen=100)
        vatdq = deque(maxlen=100)

        for i,data_point in enumerate(tig):
            if i<10:
                model.train()
                optimizer.zero_grad()
                input, target = data_point
                # if input_long:
                #     input=input.long()
                # else:
                #     input=input.float()
                # target=target.long()
                if len(target.shape) > 1:
                    target = target.squeeze(1)

                input = input.cuda()
                target = target.cuda()

                all_loss, lml, lat, lem, lvat, y = model.one_pass(input, target)
                all_loss.backward()

                # y=model(input)
                # loss=cross_entropy(y,target)
                # loss.backward()
                optimizer.step()

                pred_prob = F.softmax(y, dim=1)
                pred = torch.argmax(pred_prob, dim=1)
                hit_percentage = torch.sum(pred == target).item() / target.shape[0]

                dq.appendleft(float(all_loss))
                mldq.appendleft(float(lml.item()))
                atdq.appendleft(float(lat.item()))
                emdq.appendleft(float(lem.item()))
                vatdq.appendleft(float(lvat.item()))

        self.assertEqual(np.mean(dq),np.mean(dq))
        self.assertEqual(np.mean(mldq),np.mean(mldq))
        self.assertEqual(np.mean(atdq),np.mean(atdq))
        self.assertEqual(np.mean(emdq),np.mean(emdq))
        self.assertEqual(np.mean(vatdq),np.mean(vatdq))
