--[[/**************************************************************************
 ReVal - A Simple and Effective Machine Translation Evaluation Metric Based on Recurrent Neural Networks.

 Copyright (C) 2014 Rohit Gupta, University of Wolverhampton

 This file is part of ReVal and is a modified version of the code distributed at https://github.com/stanfordnlp/treelstm.

 ReVal is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 ReVal is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <http://www.gnu.org/licenses/>.
 **************************************************************************/
--]]
--[[

  Functions for loading data from disk.

--]]

function treelstm.read_embedding(vocab_path, emb_path)
  local vocab = treelstm.Vocab(vocab_path)
  local embedding = torch.load(emb_path)
  return vocab, embedding
end

function treelstm.read_sentences(path, vocab)
  local sentences = {}
  local file = io.open(path, 'r')
  local line
  local i = 0
  while true do
    line = file:read()
    if line == nil then break end
    local tokens = stringx.split(line)
    local len = #tokens
    local sent = torch.IntTensor(len)
    for i = 1, len do
      local token = tokens[i]
      sent[i] = vocab:index(token)
    end
    sentences[#sentences + 1] = sent
    if i % 2000 == 0 then collectgarbage() print(i) end
    i = i + 1
  end

  file:close()
  return sentences
end

function treelstm.read_trees(parent_path, label_path)
  local parent_file = io.open(parent_path, 'r')
  local label_file
  if label_path ~= nil then label_file = io.open(label_path, 'r') end
  local count = 0
  local trees = {}

  while true do
    local parents = parent_file:read()
    if parents == nil then break end
    parents = stringx.split(parents)
    for i, p in ipairs(parents) do
      parents[i] = tonumber(p)
    end

    local labels
    if label_file ~= nil then
      labels = stringx.split(label_file:read())
      for i, l in ipairs(labels) do
        labels[i] = tonumber(l)
      end
    end

    count = count + 1
    trees[count] = treelstm.read_tree(parents, labels)
    if count % 1000 == 0 then collectgarbage() print(count) end
  end
  parent_file:close()
  return trees
end

function treelstm.read_tree(parents, labels)
  local size = #parents
  local trees = {}
  if labels == nil then labels = {} end
  local root
  for i = 1, size do
    if not trees[i] and parents[i] ~= -1 then
      local idx = i
      local prev = nil
      while true do
        local parent = parents[idx]
        local tree = treelstm.Tree()
        if prev ~= nil then
          tree:add_child(prev)
        end
        trees[idx] = tree
        tree.idx = idx
        tree.gold_label = labels[idx]
        if trees[parent] ~= nil then
          trees[parent]:add_child(tree)
          break
        elseif parent == 0 then
          root = tree
          break
        else
          prev = tree
          idx = parent
        end
      end
    end
  end

  -- index leaves
  local leaf_idx = 1
  for i = 1, size do
    if trees[i].num_children == 0 then
      trees[i].leaf_idx = leaf_idx
      leaf_idx = leaf_idx + 1
    end
  end
  return root
end

--[[

  Read Training Data

--]]

function treelstm.read_relatedness_dataset(dir, vocab)
  local dataset = {}
  dataset.vocab = vocab
  dataset.ltrees = treelstm.read_trees(dir .. 'a.parents')
  dataset.rtrees = treelstm.read_trees(dir .. 'b.parents')
  dataset.lsents = treelstm.read_sentences(dir .. 'a.toks', vocab)
  dataset.rsents = treelstm.read_sentences(dir .. 'b.toks', vocab)
  dataset.size = #dataset.ltrees
  local id_file = torch.DiskFile(dir .. 'id.txt')
  local sim_file = torch.DiskFile(dir .. 'sim.txt')
  dataset.ids = torch.IntTensor(dataset.size)
  dataset.labels = torch.Tensor(dataset.size)
  for i = 1, dataset.size do
    dataset.ids[i] = id_file:readInt()
    dataset.labels[i] = 0.25 * (sim_file:readDouble() - 1)
    if i % 2000 == 0 then collectgarbage() print(i) end
  end
  id_file:close()
  sim_file:close()
  return dataset
end

--[[
  Read Test Data

--]]

function treelstm.read_test_dataset(dir, vocab)
  local dataset = {}
  dataset.vocab = vocab
  dataset.ltrees = treelstm.read_trees(dir .. 'a.parents')
  dataset.rtrees = treelstm.read_trees(dir .. 'b.parents')
  dataset.lsents = treelstm.read_sentences(dir .. 'a.toks', vocab)
  dataset.rsents = treelstm.read_sentences(dir .. 'b.toks', vocab)
  dataset.size = #dataset.ltrees
  return dataset
end

