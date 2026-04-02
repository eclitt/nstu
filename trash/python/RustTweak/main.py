import flet as ft


def main(page: ft.Page):
    page.title = "RustTweak"
    page.window_width = 800
    page.window_height = 600

    # Состояние панели
    sidebar_open = True

    # Кнопка переключения
    toggle_button = ft.IconButton(
        icon=ft.Icons.MENU,
        icon_size=24,
        rotate=0,
        on_click=lambda e: toggle_sidebar(e),
    )

    # Основной контент
    main_content = ft.Container(
        expand=True,
        padding=20,
        content=ft.Column(
            controls=[
                toggle_button,
                ft.Text("Добро пожаловать!", size=24, weight=ft.FontWeight.BOLD),
                ft.Text("Выберите пункт в боковой панели."),
            ],
        ),
    )
    settings_content = ft.Container(
        expand=True,
        padding=20,
        content=ft.Column(
            controls=[
                toggle_button,
                ft.Text("Вы в настройках", size=24, weight=ft.FontWeight.BOLD),
            ],
        ),
    )

    info_content = ft.Container(
        expand=True,
        padding=20,
        content=ft.Column(
            controls=[
                toggle_button,
                ft.Text("Вы в INFO", size=24, weight=ft.FontWeight.BOLD),
            ],
        ),
    )

    # Контейнер для динамического контента
    content_area = ft.Container(expand=True, content=main_content)

    # Меню с текстом
    menu_with_text = ft.Column(
        controls=[
            ft.Text("Меню", size=20, weight=ft.FontWeight.BOLD),
            ft.Divider(),
            ft.ListTile(
                leading=ft.Icon(ft.Icons.HOME),
                title=ft.Text("Главная"),
                on_click=lambda e: change_content(main_content),
            ),
            ft.ListTile(
                leading=ft.Icon(ft.Icons.SETTINGS),
                title=ft.Text("Настройки"),
                on_click=lambda e: change_content(settings_content),
            ),
            ft.ListTile(
                leading=ft.Icon(ft.Icons.INFO),
                title=ft.Text("О программе"),
                on_click=lambda e: change_content(info_content),
            ),
        ],
        spacing=0,
        visible=True,
    )

    # Меню без текста (только иконки)
    menu_icons_only = ft.Column(
        controls=[
            ft.Divider(),
            ft.ListTile(
                leading=ft.Icon(ft.Icons.HOME),
                on_click=lambda e: change_content(main_content),
            ),
            ft.ListTile(
                leading=ft.Icon(ft.Icons.SETTINGS),
                on_click=lambda e: change_content(settings_content),
            ),
            ft.ListTile(
                leading=ft.Icon(ft.Icons.INFO),
                on_click=lambda e: change_content(info_content),
            ),
        ],
        spacing=0,
        visible=False,
    )

    # Боковая панель
    sidebar = ft.Container(
        width=200,
        bgcolor=ft.Colors.BLACK_12,
        padding=10,
        content=ft.Column(
            controls=[menu_with_text, menu_icons_only],
            spacing=0,
        ),
    )

    # Разметка
    row = ft.Row(
        expand=True,
        controls=[
            sidebar,
            ft.VerticalDivider(width=1),
            content_area,
        ],
    )

    def change_content(new_content: ft.Container):
        content_area.content = new_content
        page.update()

    def toggle_sidebar(e):
        nonlocal sidebar_open
        sidebar_open = not sidebar_open
        if sidebar_open:
            sidebar.width = 210
            toggle_button.rotate = 0
            menu_with_text.visible = True
            menu_icons_only.visible = False
        else:
            sidebar.width = 70
            toggle_button.rotate = 1.57
            menu_with_text.visible = False
            menu_icons_only.visible = True
        page.update()

    page.add(row)


ft.run(main)
