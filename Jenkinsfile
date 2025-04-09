pipeline {
    agent any

    stages {
        stage('Clone Repository') {
            steps {
                echo 'Cloning Repository'
                git branch: 'test', 
                credentialsId: 'access-gitea',
                url: 'https://github.com/snuailab/waffle_app.git'
            }
        }
    }
}
