"""
Preprocessing script for training data.

"""
import argparse
import sys
import os
import glob
import subprocess

parser=argparse.ArgumentParser(add_help=False)
group=parser.add_argument_group('Required arguments:')
group.add_argument("-t","--training",help="training file containing tab separated segment pairs with similarity labels per line",type=str,action="store")
group.add_argument("-d","--development",help="development file",type=str,action="store")
group2=parser.add_argument_group('Other arguments:')
group2.add_argument("-h","--help",help="print this message and exit",type=str,action="store")
args=parser.parse_args()
if args.training:
	print("Training file:"+args.training)
if args.development:
	print("Development file:"+args.development)
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

def split(filepath, dst_dir):
    with open(filepath) as datafile, \
         open(os.path.join(dst_dir, 'a.txt'), 'w') as afile, \
         open(os.path.join(dst_dir, 'b.txt'), 'w') as bfile,  \
         open(os.path.join(dst_dir, 'id.txt'), 'w') as idfile, \
         open(os.path.join(dst_dir, 'sim.txt'), 'w') as simfile:
            datafile.readline()
            for line in datafile:
                i, a, b, sim, ent = line.strip().split('\t')
                idfile.write(i + '\n')
                afile.write(a + '\n')
                bfile.write(b + '\n')
                simfile.write(sim + '\n')

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

    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_dir = os.path.join(base_dir, 'training')
    lib_dir = os.path.join(base_dir, 'lib')
    train_dir = os.path.join(data_dir, 'train')
    dev_dir = os.path.join(data_dir, 'dev')
    make_dirs([train_dir, dev_dir])

    # java classpath for calling Stanford parser
    classpath = ':'.join([
        lib_dir,
        os.path.join(lib_dir, 'stanford-parser/stanford-parser.jar'),
        os.path.join(lib_dir, 'stanford-parser/stanford-parser-3.5.1-models.jar')])
    javac(os.path.join(lib_dir, 'DependencyParse.java'), cp=classpath)
    javac(os.path.join(lib_dir, 'CollapseUnaryTransformer.java'), cp=classpath)
    javac(os.path.join(lib_dir, 'ConstituencyParse.java'), cp=classpath)

    split(args.training,train_dir)
    split(args.development,dev_dir)
    # parse sentences
    parse(train_dir, cp=classpath)
    parse(dev_dir, cp=classpath)
    
    # get train and development vocabulary
    
    build_vocab(
        glob.glob(os.path.join(data_dir, 'train/*.toks'))+glob.glob(os.path.join(data_dir, 'dev/*.toks')),
        os.path.join(data_dir, 'trainvocab.txt'))
    build_vocab(
        glob.glob(os.path.join(data_dir, 'train/*.toks'))+glob.glob(os.path.join(data_dir, 'dev/*.toks')),
        os.path.join(data_dir, 'trainvocab-cased.txt'),
        lowercase=False)
