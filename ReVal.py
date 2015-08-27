'''/**************************************************************************
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
'''
"""
Reval: Main evaluation script

"""
import argparse
import sys
import os
import glob
import subprocess

parser=argparse.ArgumentParser(add_help=False)
group=parser.add_argument_group('Required arguments:')
group.add_argument("-t","--translation",help="translation file containing one segment per line",type=str,action="store")
group.add_argument("-r","--reference",help="corresponding reference file containing one segment per line",type=str,action="store")
group2=parser.add_argument_group('Other arguments:')
group2.add_argument("-h","--help",help="print this message and exit",type=str,action="store")
args=parser.parse_args()
if args.translation:
	print("Translation file:"+args.translation)
if args.reference:
	print("Reference file:"+args.reference)
if args.help:
	parser.print_help()

def make_dirs(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

def javac(filepath, cp):
    cmd = 'javac -cp %s %s' % (cp, filepath)
    print(cmd)
    os.system(cmd)

def dependency_parse(filepath, cp='', tokenize=True):
    print('\nDependency parsing ' + filepath)
    dirpath = os.path.dirname(filepath)
    filepre = os.path.splitext(os.path.basename(filepath))[0]
    tokpath = os.path.join(dirpath, filepre + '.toks')
    parentpath = os.path.join(dirpath, filepre + '.parents')
    relpath =  os.path.join(dirpath, filepre + '.rels')
    tokenize_flag = '-tokenize - ' if tokenize else ''
    cmd = ('java -cp %s DependencyParse -tokpath %s -parentpath %s -relpath %s %s < %s'
        % (cp, tokpath, parentpath, relpath, tokenize_flag, filepath))
    os.system(cmd)

def constituency_parse(filepath, cp='', tokenize=True):
    dirpath = os.path.dirname(filepath)
    filepre = os.path.splitext(os.path.basename(filepath))[0]
    tokpath = os.path.join(dirpath, filepre + '.toks')
    parentpath = os.path.join(dirpath, filepre + '.cparents')
    tokenize_flag = '-tokenize - ' if tokenize else ''
    cmd = ('java -cp %s ConstituencyParse -tokpath %s -parentpath %s %s < %s'
        % (cp, tokpath, parentpath, tokenize_flag, filepath))
    os.system(cmd)

def build_vocab(filepaths, dst_path, lowercase=True):
    vocab = set()
    for filepath in filepaths:
	print 'vocab from file:'+filepath
        with open(filepath) as f:
            for line in f:
                if lowercase:
                    line = line.lower()
                vocab |= set(line.split())
    with open(dst_path, 'w') as f:
        for w in sorted(vocab):
            f.write(w + '\n')

def processRefTra(reffilepath,trafilepath, dst_dir):
    rc=0
    tc=0
    with open(reffilepath) as datafile, \
         open(os.path.join(dst_dir, 'a.txt'), 'w') as afile:
            for line in datafile:
                a = line.strip()
                afile.write(a + '\n')
                rc=rc+1
    with open(trafilepath) as datafile, \
         open(os.path.join(dst_dir, 'b.txt'), 'w') as bfile:
            for line in datafile:
                b = line.strip()
                bfile.write(b + '\n')
                tc=tc+1
    if rc!=tc:
	print >>sys.error,"",reffilepath,"and",trafilepath,"size",differs
	sys.exit(1)

def parse(dirpath, cp=''):
    dependency_parse(os.path.join(dirpath, 'a.txt'), cp=cp, tokenize=True)
    dependency_parse(os.path.join(dirpath, 'b.txt'), cp=cp, tokenize=True)

if __name__ == '__main__':
    print('=' * 80)
    print('Preprocessing WMT Similarity dataset')
    print('=' * 80)

    base_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(base_dir, 'tmp')
    lib_dir = os.path.join(base_dir, 'lib')
    test_dir = os.path.join(data_dir, 'test')
    make_dirs([test_dir])

    # java classpath for calling Stanford parser
    classpath = ':'.join([
        lib_dir,
        os.path.join(lib_dir, 'stanford-parser/stanford-parser.jar'),
        os.path.join(lib_dir, 'stanford-parser/stanford-parser-3.5.1-models.jar')])
    javac(os.path.join(lib_dir, 'DependencyParse.java'), cp=classpath)
    javac(os.path.join(lib_dir, 'CollapseUnaryTransformer.java'), cp=classpath)
    javac(os.path.join(lib_dir, 'ConstituencyParse.java'), cp=classpath)

    processRefTra(args.reference,args.translation,test_dir)
    parse(test_dir, cp=classpath)
    
    # get test vocabulary
    build_vocab(
        glob.glob(os.path.join(data_dir, 'test/*.toks')),
        os.path.join(data_dir, 'testvocab.txt'))
    build_vocab(
        glob.glob(os.path.join(data_dir, 'test/*.toks')),
        os.path.join(data_dir, 'testvocab-cased.txt'),
        lowercase=False)

    p= subprocess.Popen('th Evaluate.lua',shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines=p.stdout.readlines()
    print lines[-1]
