import sys

def main():
    with open('/Users/harsh/Desktop/TOC/pre3/tools/trace_viewer.html', 'r', encoding='utf-8') as f:
        content = f.read()

    new_html_end_idx = content.find('    /* ----------- step model ------------------------------------------- */')

    with open('new_ui.txt', 'r', encoding='utf-8') as f:
        new_ui = f.read()

    new_content = new_ui + content[new_html_end_idx:]

    with open('/Users/harsh/Desktop/TOC/pre3/tools/trace_viewer.html', 'w', encoding='utf-8') as f:
        f.write(new_content)

if __name__ == '__main__':
    main()
