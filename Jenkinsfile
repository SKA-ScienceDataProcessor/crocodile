#!/usr/bin/env groovy

pipeline {
    agent {label 'sdp-ci-01'}
    // Master branch gets built daily, otherwise only after GitHub push
    triggers {
        cron(env.BRANCH_NAME == 'master' ? '@daily' : '')
        githubPush()
    }
    options { timestamps() }
    stages {
        stage('Test benchmark') {
            steps {
                sh '''
cd $WORKSPACE/src

make -k -j 4 test_recombine test_config iotest
./test_recombine

# Standard test
mpirun -n 2 ./iotest --rec-set=T05 | tee iotest.out
if grep ERROR iotest.out; then exit 1; fi

# Distributed tests
for i in $(seq 0 16); do
  mpirun -n 16 ./iotest --rec-set=T05 --facet-workers=$i > iotest$i.out
  if grep ERROR iotest$i.out; then exit 1; fi
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
