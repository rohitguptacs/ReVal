--[[

  ReVal:MT Evalauation script

--]]

require('.')

-- Pearson correlation
function pearson(x, y)
  x = x - x:mean()
  y = y - y:mean()
  return x:dot(y) / (x:norm() * y:norm())
end

-- read command line arguments
local args = lapp [[
  -m,--model  (default dependency) Model architecture: [dependency, lstm, bilstm]
  -l,--layers (default 1)          Number of layers (ignored for Tree-LSTM)
  -d,--dim    (default 150)        LSTM memory dimension
]]

local model_class = treelstm.TreeLSTMSim
-- directory containing dataset files
local data_dir = 'tmp/'

-- load embeddings
print('loading word embeddings')
local emb_dir = 'glove/'
local emb_prefix = emb_dir .. 'glove.840B'
--]]
--[[local train_start = sys.clock()
--]]
local emb_vocab, emb_vecs = treelstm.read_embedding(emb_prefix .. '.vocab', emb_prefix .. '.300d.th')
local testvocab = treelstm.Vocab(data_dir .. 'testvocab-cased.txt')
-- use only vectors in vocabulary (not necessary, but gives faster training)
local emb_dim = emb_vecs:size(2) --300
local num_unk = 0
local vecs = torch.Tensor(testvocab.size, emb_dim)
for i = 1, testvocab.size do
  local w = testvocab:token(i)
  if emb_vocab:contains(w) then
    vecs[i] = emb_vecs[emb_vocab:index(w)]
  else
    num_unk = num_unk + 1
    vecs[i]:uniform(-0.05, 0.05)
  end
end
print('unk count test= ' .. num_unk)
---initialze again emb_vecs with test embeddings
local emb_vecs = vecs
local test_dir = data_dir .. 'test/'
local test_dataset = treelstm.read_test_dataset(test_dir, testvocab)
-- loading a pre-trained model
local model_save_path = 'trained_model/rel-dependency.1l.150d.th'
local best_dev_model = model_class.load(model_save_path)
print 'Loaded model for testing have the following configuration'
best_dev_model:print_config()
-- evaluate
header('Evaluating on test set')
local test_predictions = best_dev_model:predict_dataset(test_dataset, emb_vecs)
-- write segment level scores
if lfs.attributes(treelstm.predictions_dir) == nil then
  lfs.mkdir(treelstm.predictions_dir)
end
local predictions_save_path = string.format(
  treelstm.predictions_dir .. '/rel-%s.%dl.%dd.pred', args.model, args.layers, args.dim)
local predictions_file = torch.DiskFile(predictions_save_path, 'w')
print('writing segment level scores to ' .. predictions_save_path)
local sysscore = 0
for i = 1, test_predictions:size(1) do
  local segscore = (test_predictions[i]-1)/4
  predictions_file:writeFloat((test_predictions[i]-1)/4)
  sysscore=sysscore + segscore
end
sysscore=sysscore/test_predictions:size(1)
print('ReVal Score:'.. sysscore)
predictions_file:close()

