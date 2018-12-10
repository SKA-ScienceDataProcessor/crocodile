#!/usr/bin/env groovy

pipeline {
    agent {label 'sdp-ci-01'}
    // Master branch gets built daily, otherwise only after GitHub push
    triggers {
        cron(env.BRANCH_NAME == 'master' ? '@daily' : '')
        githubPush()
    }
    environment {
        MPLBACKEND='agg'
        CROCODILE="${env.WORKSPACE}"
    }
    options { timestamps() }
    stages {
        stage('Setup') {
            steps {
                sh '''
# Set up fresh Python virtual environment
virtualenv -p `which python3` --no-site-packages _build
. _build/bin/activate

# Install requirements
pip install -U pip setuptools
pip install -r requirements.txt
pip install pymp-pypi
jupyter nbextension enable --py widgetsnbextension
pip install pytest pytest-xdist pytest-cov
'''
            }
        }
        stage('Test Python') {
            steps {
                sh '''
cd $WORKSPACE
. _build/bin/activate

py.test -n 4 --verbose tests
'''
            }
        }

        stage('Test Recombination') {
            steps {
                sh '''
cd $WORKSPACE/examples/grid

make -k -j 4 test_recombine test_config recombine
./test_recombine

# Standard test
mpirun -n 2 ./recombine --rec-set=T05 | tee recombine.out
if grep ERROR recombine.out; then exit 1; fi

# Distributed tests
for i in $(seq 0 16); do
  mpirun -n 16 ./recombine --rec-set=T05 --facet-workers=$i > recombine$i.out
  if grep ERROR recombine$i.out; then exit 1; fi
done
'''
           }
        }

        stage('Make Documentation') {
            steps {
                sh '''
cd $WORKSPACE
. $WORKSPACE/_build/bin/activate

make -k -j 4 -C docs html
'''
           }
        }
    }
    post {
        failure {
            emailext attachLog: true, body: '$DEFAULT_CONTENT',
                recipientProviders: [culprits()],
                subject: '$DEFAULT_SUBJECT',
                to: '$DEFAULT_RECIPIENTS'
        }
        fixed {
            emailext body: '$DEFAULT_CONTENT',
                recipientProviders: [culprits()],
                subject: '$DEFAULT_SUBJECT',
                to: '$DEFAULT_RECIPIENTS'
        }
    }
}
