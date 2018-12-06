#!/usr/bin/env groovy

pipeline {
    agent {label 'sdp-ci-01'}
    environment {
        MPLBACKEND='agg'
        CROCODILE="${env.WORKSPACE}"
    }
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
        stage('Test') {
            steps {
                sh '''
cd $WORKSPACE
. _build/bin/activate

py.test -n 4 --verbose tests
'''
            }
        }

       stage('Run Notebooks') {
           steps {
               sh '''
cd $WORKSPACE
. $WORKSPACE/_build/bin/activate

make -C docs html
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
