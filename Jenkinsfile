pipeline {
    agent any

    stages {
        stage('Clone Repository') {
            steps {
                // 깃 플러그인을 이용한 클론 방법
                // URL과 branch를 자신의 저장소에 맞게 변경하세요.
                git branch: 'main', url: 'https://github.com/Sangh0/MLOps-Tutorial.git'
            }
        }
        stage('Test Stage') {
            steps {
                // 클론된 레포지토리 안에서 간단한 테스트 실행 (예: 파일 목록 확인)
                sh 'echo "레포지토리 클론 성공, 테스트를 실행합니다."'
                sh 'ls -la'
            }
        }
    }
}
