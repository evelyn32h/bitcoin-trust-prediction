#!/usr/bin/env python3
"""
一键修复所有Unicode编码问题的脚本
解决Windows GBK编码环境下的Unicode字符显示问题
"""

import os
import re
from pathlib import Path

def fix_unicode_in_file(file_path, replacements):
    """修复单个文件中的Unicode字符"""
    try:
        # 读取文件
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # 应用替换
        original_content = content
        for old_char, new_char in replacements.items():
            content = content.replace(old_char, new_char)
        
        # 如果有修改，写回文件
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"❌ 处理文件 {file_path} 时出错: {e}")
        return False

def main():
    print("🔧 UNICODE字符修复工具")
    print("=" * 50)
    
    # Unicode字符替换映射
    unicode_replacements = {
        # 基本符号
        '\u2713': 'SUCCESS',  # ✓
        '\u2717': 'FAILED',   # ✗ 
        '\u2715': 'FAILED',   # ✕
        
        # 彩色圆点 (emoji)
        '\U0001f7e2': '[GREEN]',   # 🟢
        '\U0001f7e0': '[YELLOW]',  # 🟠
        '\U0001f534': '[RED]',     # 🔴
        
        # 其他可能的Unicode字符
        '✓': 'SUCCESS',
        '✗': 'FAILED',
        '✕': 'FAILED',
        '🟢': '[GREEN]',
        '🟠': '[YELLOW]',
        '🔴': '[RED]',
    }
    
    # 需要修复的文件列表 (根据实际文件位置)
    files_to_fix = [
        'run_experiment.py',        # 根目录下
        'train_model.py',           # 根目录下
        'validate_model.py',        # 根目录下
        'test_model.py',            # 根目录下
        'src/utilities.py',         # src目录下
        'run_batch_experiments.py', # 根目录下
    ]
    
    fixed_count = 0
    total_count = 0
    
    for file_path in files_to_fix:
        if os.path.exists(file_path):
            total_count += 1
            print(f"🔍 检查文件: {file_path}")
            
            if fix_unicode_in_file(file_path, unicode_replacements):
                print(f"  ✅ 修复完成")
                fixed_count += 1
            else:
                print(f"  ℹ️  无需修复")
        else:
            print(f"  ⚠️  文件不存在: {file_path}")
    
    print("\n" + "=" * 50)
    print(f"📊 修复总结:")
    print(f"  检查文件数: {total_count}")
    print(f"  修复文件数: {fixed_count}")
    
    if fixed_count > 0:
        print(f"\n🎉 修复完成! 现在可以重新运行实验了:")
        print(f"   python run_batch_experiments.py")
    else:
        print(f"\n📝 所有文件都是最新的，无需修复")
    
    # 额外检查：查找可能包含Unicode字符的其他Python文件
    print(f"\n🔍 扫描其他可能的问题文件...")
    unicode_pattern = re.compile(r'[\u2700-\u27BF]|[\uE000-\uF8FF]|[\u2600-\u26FF]|[\u2713\u2717\u2715]|[\U0001F600-\U0001F64F]|[\U0001F300-\U0001F5FF]|[\U0001F680-\U0001F6FF]|[\U0001F700-\U0001F77F]|[\U0001F780-\U0001F7FF]|[\U0001F800-\U0001F8FF]')
    
    found_issues = []
    for py_file in Path('.').rglob('*.py'):
        try:
            with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                if unicode_pattern.search(content):
                    found_issues.append(str(py_file))
        except:
            continue
    
    if found_issues:
        print(f"  发现可能包含Unicode字符的其他文件:")
        for file in found_issues:
            print(f"    - {file}")
        print(f"  建议手动检查这些文件")
    else:
        print(f"  ✅ 未发现其他问题文件")

if __name__ == "__main__":
    main()