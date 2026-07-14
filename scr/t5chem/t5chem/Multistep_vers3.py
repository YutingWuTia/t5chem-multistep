
import argparse
import os
from functools import partial
from typing import Dict, List, NamedTuple
import pdb
import torch
from t5chem import EarlyStopTrainer, SimpleTokenizer,T5ForProperty
from t5chem.data_utils import AccuracyMetrics, MultiStepForwardDataset
from t5chem.model import T5ForMultiStepBase
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import (BatchEncoding, T5Config, TrainingArguments)
from transformers.modeling_outputs import Seq2SeqLMOutput


class T5ForMultiStep(T5ForMultiStepBase):
    r"""
    Multi-step T5Chem Model. 
    Args:
        config (:obj:`T5Config`):
            The configuration of T5Chem 
    """
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # make attention_mask, labels and decoder_input_ids to be the last step...
        this_input = input_ids[0]
        if len(input_ids) == 1:        # if the whole batch are all single step reaction.
            attention_mask = attention_mask[0]
        else:
            for i in range(len(input_ids)-1):
                inputs = generate_step(self.generator, this_input, input_ids[i+1])
                this_input = inputs['input_ids'].to(self.device)
                attention_mask = inputs['attention_mask'].to(self.device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.generator.encoder(
                input_ids=this_input,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        hidden_states = encoder_outputs[0]

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
                # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

            # If decoding with past key value states, only the last tokens
            # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Decode
        decoder_outputs = self.generator.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.generator.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
    
    def generate(self, input_ids,attention_mask, **kwargs):
        with torch.no_grad():
            this_input = input_ids[0]
        if len(input_ids) == 1:        # if the whole batch are all single step reaction.
            attention_mask = attention_mask[0]
        else:
            for i in range(len(input_ids)-1):
                inputs = generate_step(self.generator, this_input, input_ids[i+1])
                this_input = inputs['input_ids'].to(self.device)
                attention_mask = inputs['attention_mask'].to(self.device)

        outputs = model.generator.generate(
                input_ids=this_input, 
                attention_mask=attention_mask,
                early_stopping=True,
                max_length=150,
                num_beams=5,
                num_return_sequences=5,
                decoder_start_token_id=3,
            )
        return outputs

def data_collatorForNStep(batch: List[Dict[str, List[int]]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    whole_batch: Dict[str, torch.Tensor] = {}
    # Padding the reaction steps with empty list.
    # get the exsample with longest reaction steps:
    index_max_steps, max_number_of_steps = max(enumerate(len(x["input_ids"]) for x in batch), \
                                                key=lambda x: x[1], default=(0, 0)) 
    
    for i in range(len(batch)):
        number_of_padding_steps = max_number_of_steps - len(batch[i]["input_ids"])
        batch[i]["input_ids"] = [[]] * number_of_padding_steps + batch[i]["input_ids"] 
        batch[i]["attention_mask"] = [[]] * number_of_padding_steps + batch[i]["attention_mask"] 

    # Padding string within each reaction step
    ex: BatchEncoding = batch[index_max_steps]
    for key in ex.keys():
        whole_batch[key] = ()
        if 'mask' in key:
            padding_value = 0
        else:
            padding_value = pad_token_id
        for i in range(len(ex[key])):
            whole_batch[key] += (
                                pad_sequence([torch.LongTensor(x[key][i]) for x in batch],
                                batch_first=True,
                                padding_value=padding_value),
                                )
    source_ids, source_mask, y = \
        whole_batch["input_ids"], whole_batch["attention_mask"], whole_batch["decoder_input_ids"]
    return {'input_ids': source_ids, 'attention_mask': source_mask,
            'labels': y[0]}  
    # source_ids[the_ith_step][the_ith_sample_in_batch],same as source_mask
    # For fake step, source_ids == [3,3...3], source_mask == [0,0...0]
    # For real step, source_ids like: [85,5,7,12,..3,3,3], source_mask == [1,1...1]

tokenizer = SimpleTokenizer("/scratch/yw5806/multistep/models/USPTO_500_MT/vocab.pt")

def generate_step(model, cur_inputs, next_input):
    #get the index of fake inputs
    batch_size = cur_inputs.size(0)
    skip_index_list = [i for i in range(batch_size) if (cur_inputs[i] == 3).all()]
    calcul_index_list = [i for i in range(batch_size) if i not in skip_index_list]
    calcul_cur_inputs = cur_inputs[[i for i in range(batch_size) if i in calcul_index_list]]
    # This is the most time consuming process, generating predictions based on previous REAL inputs.   
    outputs = model.generate(calcul_cur_inputs, num_beams=5, max_length=150, early_stopping=True) 
    next_inputs = []
    for i in range(batch_size):
        if i in calcul_index_list:            
            index_in_outputs = calcul_index_list.index(i)     # index() method searches for the 1st occurrence of the specified value i in the list and returns its index.
            pred_str = tokenizer.decode(outputs[index_in_outputs], 
                                        skip_special_tokens=True, clean_up_tokenization_spaces=True)
            next_str = tokenizer.decode(next_input[i], 
                                        skip_special_tokens=True)
            next_inputs.append( next_str[: next_str.find('Product:')+ 9 ] 
                               + pred_str + "." + next_str[next_str.find('Product:') + 9 :] )        
        elif i in skip_index_list:
            next_str = tokenizer.decode(next_input[i], skip_special_tokens=True)
            next_inputs.append( next_str ) 
        else:
            print("------The index for def generate_step dosen't match: ", i)
    return tokenizer(next_inputs, padding="longest", return_tensors='pt')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/scratch/yw5806/Multistep/multistep/",
        help="Where to write checkpoints/logs and save the final model."
    )
    parser.add_argument(
        "--pretrain",
        type=str,
        default="/scratch/yw5806/multistep/models/USPTO_500_MT/",
        help="Path to the pretrained model directory."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/scratch/yw5806/Multistep/data/concatenated/",
        help="Path to the dataset directory."
    )

    args = parser.parse_args()

    output_dir = args.model_dir
    
    # testing start 
    model_dir = args.pretrain
    dict = torch.load(os.path.join( model_dir,"pytorch_model.bin"),map_location=torch.device('cpu'))
    key_name = list(dict.items())[0][0]
    if "generator" in key_name:
        model = T5ForMultiStep.from_pretrained(model_dir)
    else:
        config = T5Config.from_pretrained(os.path.join( model_dir,"config.json"))
        model = T5ForMultiStep(config)
        model0_dict = dict
        
        for k,v in list(model0_dict.items()):
            k_modified= "generator."+k
            model0_dict[k_modified] = model0_dict.pop(k)  
            # in this way we add "generator." and the order of the keys didn't change cause we modified all keys one by one.
       
        model.load_state_dict(model0_dict)
        # End of the modification of pre-trained model(USPTO_500_MT)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = MultiStepForwardDataset(
        tokenizer, 
        data_dir=args.data_dir, # /scratch/yw5806/Multistep/data/concatenated/
        type_path="train",
    )
    eval_iter = MultiStepForwardDataset(
        tokenizer, 
        data_dir=args.data_dir, # /scratch/yw5806/Multistep/data/concatenated/
        type_path="val",
    )
    data_collator_padded = partial(
                data_collatorForNStep, pad_token_id=tokenizer.pad_token_id)
    compute_metrics = AccuracyMetrics
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        do_train=True,
        evaluation_strategy='steps',
        num_train_epochs=100,
        per_device_train_batch_size=32,
        logging_steps=5,
        per_device_eval_batch_size=32,
        save_steps=50,
        save_total_limit=3,
        learning_rate=1e-4,
        prediction_loss_only=(compute_metrics is None),
    )
    # This training_args is the same as /scratch/yw5806/Multistep/FinetunedD1
    trainer = EarlyStopTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator_padded,
            train_dataset=dataset,
            eval_dataset=eval_iter,
            compute_metrics=compute_metrics,
        )
    
    trainer.train("")
    trainer.save_model(output_dir)
