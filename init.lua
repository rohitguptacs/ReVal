--[[
/**************************************************************************
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
require('torch')
require('nn')
require('nngraph')
require('optim')
require('xlua')
require('sys')
require('lfs')

treelstm = {}

include('util/read_data.lua')
include('util/Tree.lua')
include('util/Vocab.lua')
include('layers/CRowAddTable.lua')
include('models/LSTM.lua')
include('models/TreeLSTM.lua')
include('models/ChildSumTreeLSTM.lua')
include('relatedness/LSTMSim.lua')
include('relatedness/TreeLSTMSim.lua')

printf = utils.printf

-- global paths (modify if desired)
treelstm.data_dir        = 'training'
treelstm.models_dir      = 'new_trained_models'
treelstm.predictions_dir = 'results'

-- share parameters of nngraph gModule instances
function share_params(cell, src, ...)
  for i = 1, #cell.forwardnodes do
    local node = cell.forwardnodes[i]
    if node.data.module then
      node.data.module:share(src.forwardnodes[i].data.module, ...)
    end
  end
end

function header(s)
  print(string.rep('-', 80))
  print(s)
  print(string.rep('-', 80))
end
