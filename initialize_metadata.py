import os
import sys
from document_manager import DocumentManager

def initialize_metadata_for_files(file_list, data_folder):
    """Initialize metadata for a list of files"""
    manager = DocumentManager(data_folder)
    
    for file_name in file_list:
        # Extract numeric prefix and title from filename
        parts = file_name.split(' ', 1)
        number = parts[0] if parts else ""
        title = parts[1].split('.')[0] if len(parts) > 1 else file_name.split('.')[0]
        
        # Check if file exists in data folder
        file_path = os.path.join(data_folder, file_name)
        if os.path.exists(file_path):
            # Add metadata
            doc = manager.add_document(
                file_path,
                title=title,
                tags=[number] if number.isdigit() else []
            )
            print(f"Added metadata for: {file_name}")
        else:
            print(f"File not found: {file_name}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python initialize_metadata.py <data_folder>")
        return
    
    data_folder = sys.argv[1]
    
    # List of files from your screenshot
    file_list = [
        "4 Квал требований.docx",
        "5. Положение об учебно-методическом совете (редакция 3).docx",
        "Академическая политика 06122023 УС.docx",
        "47 Документ гос образ.docx",
        "84 Форма адм данных.docx",
        "122 Размещения гос заказа.docx",
        "136 Правила назначения стипендий.docx",
        "137 Дистанции и онлайн обуч.docx",
        "141 Правила размещения гос кредит.docx",
        "152 Правила кредит обуч.docx",
        "190 Правила КТ.docx",
        "204 ЕНТ.docx",
        "268 Правила призн докум.docx",
        "311 Подуше финан.docx",
        "361 Правила дуального обуч.docx",
        "403 Правила на работу.docx",
        "443 Правила присуждения бакалавр и магистр.docx",
        "449 Правила образ мониторинга.docx",
        "583 УМ и НМР.docx",
        "595 Типовые правила деятельности.docx",
        "600 Типовые правила приема.docx",
        "606 Среднее соотнош.docx",
        "629 Требований к аккред органу.docx",
        "ДОТ.doc",
        "Кодировка дичциплин.docx",
        "Методические рекомендации по организации СРС- red.docx",
        "МЕТОДИЧЕСКИЕ УКАЗАНИЯ ПО ВЫПОЛНЕНИЮ НАУЧНО-ИССЛЕДОВАТЕЛЬСКОЙ РАБОТЫ ДОКТОРАНТА.doc",
        "МУ по составлению Силлабус .docx",
        "Перевод восстановление 2024 5 редакция +.docx",
        "Планирование и учет нагрузки ППС от10.06.24.docx",
        "Положение о стипендии и соц поддержке 2022.docx",
        "Положение об организации учебно-методической деятельности.docx",
        "положение об эдвайзерстве 26.03.2024.docx",
        "ПОЛОЖЕНИЕ ПО ОРГАНИЗАЦИИ И ПРОВЕДЕНИЮ ПРАКТИКИ ОБУЧАЮЩИХСЯ ТОО «ASTANA IT UNIVERSITY».docx",
        "Положению по оцениванию 05.01.2025 обновленный.docx",
        "Правила орг и проведения промежуточной аттестации 06.12.2022.docx",
        "Правила проведения итоговой аттестации обучающихся 23.05.2024.docx",
        "Правила разработки ОП для УС ИТОГ.doc",
        "ПРАВИЛА_ПРИЗНАНИЯ_РЕЗУЛЬТАТОВ_ОБУЧЕНИЯ_25_11_2024.docx",
    ]
    
    initialize_metadata_for_files(file_list, data_folder)
    print("Metadata initialization complete.")

if __name__ == "__main__":
    main()