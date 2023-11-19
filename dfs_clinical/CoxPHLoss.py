import torch
from torch import Tensor

class CoxPHLoss(torch.nn.Module):
    def forward(self, log_h: Tensor, durations: Tensor, events: Tensor) -> Tensor:
        return self.cox_ph_loss(log_h, durations, events)

    def cox_ph_loss(self, log_h: Tensor, durations: Tensor, events: Tensor, eps: float = 1e-7) -> Tensor:
        idx = durations.sort(descending=True)[1]
        events = events[idx]
        log_h = log_h[idx]
        return self.cox_ph_loss_sorted(log_h, events, eps)

    def cox_ph_loss_sorted(self, log_h: Tensor, events: Tensor, eps: float = 1e-7) -> Tensor:
        if events.dtype is torch.bool:
            events = events.float()
        
        events = events.view(-1)
        log_h = log_h.view(-1)
        gamma = log_h.max()
        log_cumsum_h = log_h.sub(gamma).exp().cumsum(0).add(eps).log().add(gamma)
        return - log_h.sub(log_cumsum_h).mul(events).sum().div(events.sum())


def safe_normalize(x):
    """Normalize risk scores to avoid exp underflowing.

    Note that only risk scores relative to each other matter.
    If minimum risk score is negative, we shift scores so minimum
    is at zero.
    """
    x_min = torch.min(x,0)[0]
    return x-x_min


def logsumexp_masked(risk_scores,mask,axis,keepdims):
    """Compute logsumexp across `axis` for entries where `mask` is true."""
    assert len(risk_scores.shape) == len(mask.shape)

    # with tf.name_scope("logsumexp_masked"):
    mask_f = mask.type(risk_scores.dtype)
    risk_scores_masked = torch.mul(risk_scores, mask_f)
    # for numerical stability, substract the maximum value
    # before taking the exponential
    amax = torch.max(risk_scores_masked, axis=axis, keepdims=True)[0]
    risk_scores_shift = risk_scores_masked - amax

    exp_masked = torch.mul(torch.exp(risk_scores_shift),mask_f)
    exp_sum = torch.sum(exp_masked, axis=axis, keepdims=True)
    output = amax + torch.log(exp_sum)
    if not keepdims:
        output = torch.squeeze(output, axis=axis)
    return output


def coxPHLoss(output, target):
    '''
    output: The predicted outputs. Must be a rank 2 tensor.
    target: list|tuple of tf.Tensor : E (0/1), riskset 
    '''
    event, riskset = target
    predictions = output

    pred_shape = predictions.shape
    if len(pred_shape) ==1:
        predictions = torch.unsqueeze(predictions, 1)
        pred_shape = predictions.shape
    if len(pred_shape) != 2:
        raise ValueError("Rank mismatch: Rank of predictions (received %s) should "
                            "be 2." % pred_shape.ndims)

    if pred_shape[1] is None:
        raise ValueError("Last dimension of predictions must be known.")

    if pred_shape[1] != 1:
        raise ValueError("Dimension mismatch: Last dimension of predictions "
                             "(received %s) must be 1." % pred_shape[1])

    if len(event.shape)==1:
        event = torch.unsqueeze(event, 1)

    if len(event.shape) != len(pred_shape):
        raise ValueError("Rank mismatch: Rank of predictions (received %s) should "
                             "equal rank of event (received %s)" % (
                            len(pred_shape), len(event.shape)))

    if len(riskset.shape)==1:
        riskset = torch.unsqueeze(riskset, 1)
    if len(riskset.shape) != 2:
        raise ValueError("Rank mismatch: Rank of riskset (received %s) should "
                            "be 2." % len(riskset.shape))

    
    
    event = event.type(predictions.dtype)
    predictions = safe_normalize(predictions) 

    # with tf.name_scope("assertions"):
    #     assertions = (
    #         tf.debugging.assert_less_equal(event, 1.),
    #         tf.debugging.assert_greater_equal(event, 0.),
    #         tf.debugging.assert_type(riskset, tf.bool)
    #     )

    # move batch dimension to the end so predictions get broadcast
    # row-wise when multiplying by riskset
    pred_t = torch.transpose(predictions,0,1)
    # compute log of sum over risk set for each row
    rr = logsumexp_masked(pred_t, riskset, axis=1, keepdims=True)
    assert rr.shape == predictions.shape

    losses = torch.mul(event, rr - predictions)
    # print(losses)
    # print(output)
    # print(event)
    losses = torch.sum(losses) # ori

    # losses = torch.mean(losses)

    return losses