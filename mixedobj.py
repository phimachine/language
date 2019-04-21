# https://www.aaai.org/Papers/AAAI/2019/AAAI-SachanD.7236.pdf

import torch.nn as nn
import torch.nn.functional as F
import torch
from modeltran import *
import numpy as np

def norm2reci(tensor):
    norm= tensor.pow(2).sum().rsqrt()
    return norm

class AT(nn.Module):
    def __init__(self, after_embedding, epsilon=0.05):
        """
        AT adds a normalized vector upon the input embedding that is in the direction that increases loss by \epsilon
        The model is required to perform as well with the adversarial data point.

        Because the partial derivative to the loss is calculated, a pass through the original model needs to be run first

        AT does not include the CE criterion.

        embedding is of shape (batch, embedding)

        AT is generated from embedding_grad, which requires targets.
        """
        super(AT, self).__init__()
        # the after_embedding model
        # the original, not the copy. object, not a class
        self.after_embedding=after_embedding
        self.epsilon=epsilon

    def forward(self,embedding, embedding_grad):
        """

        :param embedding:
        :return: free form R
        """
        # embedding=embedding.detach()
        # embedding.requires_grad=True
        # embedding.retain_grad()
        radv=self.epsilon*embedding_grad*norm2reci(embedding_grad)

        new_embed=embedding+radv
        # new input half way in the model.
        output=self.after_embedding(new_embed)
        return output

class EM(nn.Module):
    def __init__(self):
        super(EM, self).__init__()

    def forward(self, output):
        # the output is logit
        prob=F.softmax(output, dim=1)
        logprob=F.log_softmax(output,dim=1)
        batch_loss=-torch.sum(prob*logprob, dim=1)
        loss=batch_loss.mean(dim=0)
        return loss

class VAT(nn.Module):

    """
    VAT is generated from embedding itself and not embedding, hence unsupervised.
    """
    def __init__(self, after_embedding, xi=1):
        super(VAT, self).__init__()
        self.after_embedding=after_embedding
        self.xi=xi

    def get_g(self, embedding, output):
        """
        :param embedding:
        :param output:
        :return:
        """
        # detach, because g is a noise generator, does not require grad.
        # detach returns a new tensor and does not require grad, so we make it.
        embedding=embedding.detach()
        # embedding.requires_grad=True
        # embedding.retain_grad()
        noise_sample=torch.zeros_like(embedding).normal_()
        xid=self.xi*noise_sample*norm2reci(noise_sample)
        vprimei=embedding+xid
        # none of this requires grad so far
        afterpipe=self.after_embedding(vprimei)
        beforepipe=output
        # KL divergence
        dkl=self.kl_divergence(beforepipe,afterpipe).sum()
        # this will be backed to embedding? to avoid interference, I should detach embedding
        embedding_grad=torch.autograd.grad(dkl,embedding, retain_graph=True, only_inputs=True)[0]
        return embedding_grad

    def kl_divergence(self,beforepipe, afterpipe):
        """

        :param beforepipe: free form R
        :param afterpipe: ffR, same dimension
        :return: dkl: unsumed, dimension equal to either pipe
        """
        p=F.softmax(beforepipe,dim=1)
        beforepipe=F.log_softmax(beforepipe, dim=1)
        afterpipe=F.log_softmax(afterpipe, dim=1)
        dkl=p*(beforepipe-afterpipe)
        return dkl

    def forward(self, out, vout):
        """

        :param vout: ffR, prediction
        :param out: ffR, prediction
        :return:
        """
        dkl=self.kl_divergence(out,vout).sum(dim=1).mean(dim=0)
        return dkl



class TransformerBOWMixed(nn.Module):
    # no attention module here, because the input is not timewise
    # we will add time-wise attention to time-series model later.

    def __init__(self, d_model=256, vocab_size=50000, d_inner=32, dropout=0.1, n_layers=8, output_size=12, epsilon=1, xi=1,
                 lambda_ml=1, lambda_at=1, lambda_em=1, lambda_vat=1):
        super(TransformerBOWMixed, self).__init__()


        self.d_inner=d_inner
        self.d_model=d_model
        self.n_layer=n_layers
        self.dropout=dropout

        self.vocab_size=vocab_size
        self.embedding=torch.nn.Parameter(torch.Tensor(vocab_size, d_model))
        # self.first_embedding=nn.Linear(vocab_size,d_model)
        self.last_linear = nn.Linear(d_model, output_size)
        self.layer_stack = nn.ModuleList([
            EncoderLayerBOW(d_model, d_inner, dropout=dropout)
            for _ in range(n_layers)])
        # always holds reference to a set of embedding
        # should not be a memory issue, since it's released after every training set
        # however, after training finished, unless the model is released, this tensor will remain on the GPU
        self.epsilon=epsilon

        self.lambda_ml=lambda_ml
        self.lambda_at=lambda_at
        self.lambda_em=lambda_em
        self.lambda_vat=lambda_vat
        self.xi=xi

    @staticmethod
    def reset_mod(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            m.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.embedding.data)
        self.apply(self.reset_mod)

    def one_pass(self, input, target):
        # ml
        output=self(input)
        lml = F.cross_entropy(output, target)

        # adv, at
        embed_grad = torch.autograd.grad(lml, self.embedding, only_inputs=True, retain_graph=True)[0]
        radv=self.radv(embed_grad)
        yat=self(input, radv)
        lat=F.cross_entropy(yat, target)

        # unsupervised
        lem = self.em(output)

        # vat
        xid=self.xid()
        aoutput=self(input, xid)
        rvat=self.rvat(output, aoutput)
        yvat=self(input,rvat)
        lvat=self.kl_divergence(output, yvat)
        lvat=lvat.sum(dim=1).mean(dim=0)

        all_loss=self.lambda_ml*lml+self.lambda_at*lat+self.lambda_em*lem+self.lambda_at*lvat
        return all_loss, lml, lat, lem, lvat, output


    def forward(self, input, r=None):
        """
        pass one time with embedding_grad=None

        :param input: (batch_size, embedding)
        :return:
        """
        # enc_output=self.first_embedding(input)
        if r is None:
            enc_output=torch.matmul(input, self.embedding)
        else:
            enc_output=torch.matmul(input, (self.embedding + r))

        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output)

        output=self.last_linear(enc_output)
        output=output.squeeze(1)
        return output

    def radv(self, embedding_grad):
        """
        computatino of embedding_grad should not modify the graph and the graph's grad in any way.
        pass this to the forward
        """
        radv = self.epsilon * embedding_grad * norm2reci(embedding_grad)
        return radv

    def xid(self):
        """ pass this to the forward then KLD """
        noise_sample=torch.zeros_like(self.embedding).normal_()
        xid=self.xi*noise_sample*norm2reci(noise_sample)
        return xid

    def rvat(self, output, aoutput):
        """
        this function should not incur any gradient backprop
        xid is sampled from the thin-air, passed to forward argument, so the whole model is included in the graph
        to compute dkl and g. However, the backprop is configured to not modify any gradient of the models, certainly
        not values. So when autograd takes care of its business, nothing here will be considered. To autograd,
        rvat should be just a tensor.
        Despite previous gradient passes, g should not be influenced, because derivative, well, is derivative.

        :param output: free form R
        :param aoutput: adversarial output, free form R, returned by passing xid to forward
        :return:
        """
        # embed_copy=self.embedding.detach()
        beforepipe=output
        afterpipe=aoutput
        dkl=self.kl_divergence(beforepipe, afterpipe)
        # this pass does not accumulate the gradient. since this is the only backward pass
        dkl=dkl.sum(dim=1).mean(dim=0)
        g=torch.autograd.grad(dkl, self.embedding, retain_graph=True, only_inputs=True)[0]
        rvat=self.radv(g)
        return rvat

    @staticmethod
    def em(output):
        """
        treating this function as a functional should work? I hope? I do not know for sure.
        if does not work, wrap it with nn.Module as a forward()
        :param output:
        :return:
        """
        # the output is logit
        prob=F.softmax(output, dim=1)
        logprob=F.log_softmax(output,dim=1)
        batch_loss=-torch.sum(prob*logprob, dim=1)
        loss=batch_loss.mean(dim=0)
        return loss

    @staticmethod
    def kl_divergence(beforepipe, afterpipe):
        """

        :param beforepipe: free form R
        :param afterpipe: ffR, same dimension
        :return: dkl: unsumed, dimension equal to either pipe
        """
        p = F.softmax(beforepipe, dim=1)
        beforepipe = F.log_softmax(beforepipe, dim=1)
        afterpipe = F.log_softmax(afterpipe, dim=1)
        dkl = p * (beforepipe - afterpipe)
        return dkl


#
#
# class TransformerBOWAfterEmbedding(nn.Module):
#     def __init__(self, d_inner=32, dropout=0.1, d_model=256, n_layers=8, output_size=12):
#         super(TransformerBOWAfterEmbedding, self).__init__()

#
#
#     @staticmethod
#     def reset_mod(m):
#         classname = m.__class__.__name__
#         if classname.find('Linear') != -1:
#             m.reset_parameters()
#
#     def reset_parameters(self):
#         self.apply(self.reset_mod)

    #
    # def forward(self, input):
    #     """
    #
    #     :param input: (batch_size, embedding)
    #     :return:
    #     """


def weight_gradient_test():
    a=torch.Tensor([1,2,3,4])
    b=torch.Tensor([2,3,4,5])

    w=torch.Tensor([2])
    w.requires_grad=True
    wa=torch.sum(a*w)

    grad=torch.autograd.grad(wa, w, retain_graph=True, only_inputs=True)
    print(grad)
    print(w)
    print(w.grad)

    wb=torch.sum(b*w)

    grad=torch.autograd.grad(wb, w, retain_graph=True, only_inputs=True)
    print(grad)
    print(w)
    print(w.grad)

    print("Done")




def gradient_test():
    a=torch.Tensor([1,2,3,4])
    a.requires_grad=True
    b=a*4
    c=torch.sum(b*8)

    c.backward()
    print(a.grad)
    print(b.grad)

    # the question is whether the gradient accumulates

    # original
    a2 = torch.Tensor([1, 2, 3, 4])
    a2.requires_grad = True
    b2=a2*4
    d2 = b2*12
    d2=torch.sum(d2*2)
    d2.backward()


    print(a2.grad)
    print(b2.grad)

    # stacked
    a3=a.detach()

    print(a.grad)
    print(a3.grad)
    b=a3*4
    d=b*12
    d=torch.sum(d*2)
    d.backward()

    # stack properly
    print(a.grad)
    print(b.grad)
    
def multi_pass_test():
    # test the .after_embedding() behavior during multiple passes
    class testModule(nn.Module):
        def __init__(self):
            super(testModule, self).__init__()
            self.weight=torch.nn.Parameter(torch.Tensor([1]))

        def forward(self,input):
            return self.weight*input

    input1=torch.Tensor([1,2,3])
    input2=torch.Tensor([5,6,7])
    t1=testModule()
    t2=testModule()
    t3=testModule()

    l1=t1(input1).sum()
    l1.backward()
    print(t1.weight.grad)

    l2=t2(input2).sum()
    l2.backward()
    print(t2.weight.grad)

    l3_1=t3(input1).sum()
    l3_2=t3(input2).sum()
    l3=l3_1+l3_2
    l3.backward()
    print(t3.weight.grad)


def gradient_accumulation_test():
    """
    see if parameters behave as expected.
    :return:
    """
    a = torch.Tensor([1, 2, 3, 4])
    a.requires_grad = True
    b = a * 4
    c = torch.sum(b * 8)

    grad=torch.autograd.grad(c,a,retain_graph=True,only_inputs=True)
    print(grad)
    print(a.grad)

def one_pass_test():
    class Dummydumb(nn.Module):
        def __init__(self):
            super(Dummydumb, self).__init__()
            self.line=torch.nn.Parameter(torch.ones(500,12))

        def forward(self, input):
            """

            :param input:
            :return:
            """
            return torch.matmul(input, self.line)

    vocab_size=500
    maxlen=100
    bs=37

    model=TransformerBOWMixed(vocab_size=vocab_size)
    # model=Dummydumb()
    em=EM()
    at=AT(model.after_embedding)
    vat=VAT(model.after_embedding)

    # some dummy
    input=np.random.uniform(size=vocab_size*bs).reshape((bs, vocab_size))
    input=torch.Tensor(input).float()
    target=torch.from_numpy(np.random.randint(0,12,size=bs)).long()

    onepass(model,input, target, em, at, vat)

#
# def new_onepass(model, input, target, em, at, vat):
#     y = model(input)
#     lml = F.cross_entropy(y, target)
#     # to get the correct gradient, lambda term is not considered.
#     embedding=model.embedding
#     # this generates embedding gradients
#     embed_grad=torch.autograd.grad(lml,embedding, retain_graph=True)[0]
#
#

# how to train this model?
def onepass(model, input, target, em, at, vat):

    # supervised

    y = model(input)
    lml = F.cross_entropy(y, target)
    # to get the correct gradient, lambda term is not considered.
    embedding=model.embedding
    # this generates embedding gradients
    embed_grad=torch.autograd.grad(lml,embedding, retain_graph=True)[0]

    # grad does not require grad by default

    # everything from now on detaches without grad
    yat=model(input, embed_grad)
    lat=F.cross_entropy(yat, target)

    # unsupervised
    # em requires absolute predictions
    lem=em(y)

    # this tensor does not require grad. it is itself a leaf. embedding has been detached
    # embedding and y has no information on the target, thus unsupervised
    gvat=vat.get_g(embedding,y)
    # two passes through the same model should not interfere with each other?
    # there is no guarantee? I am not sure
    # every pass on the module should be saved, otherwise models such as RNN or LSTM will lose their past weights,
    # and the gard calculation relies on them. I do not worry.
    yvat=at(embedding,gvat)
    lvat=vat(y, yvat)
    
    
    lambda_ml=1
    lambda_at=1
    lambda_em=1
    lambda_vat=1

    #   allloss=lambda_ml*lml+lambda_at*lat+lambda_em*lem+lambda_vat*lvat
    allloss=lambda_ml*lml+lambda_em*lem

    return lml.item(), lat.item(), lem.item(), lvat.item(), allloss.item(), allloss, y

def test_modular_one_pass():

    vocab_size=500
    bs=37

    model=TransformerBOWMixed(vocab_size=vocab_size)

    # some dummy
    input=np.random.uniform(size=vocab_size*bs).reshape((bs, vocab_size))
    input=torch.Tensor(input).float()
    target=torch.from_numpy(np.random.randint(0,12,size=bs)).long()

    loss=model.one_pass(input, target)
    loss.backward()
    print(loss)

if __name__ == '__main__':
    test_modular_one_pass()