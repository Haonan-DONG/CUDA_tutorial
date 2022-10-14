#include <stdio.h>
#include <stdlib.h>

void test1()
{
    int b = 3;
    int *a = &b;
    *a = *a + 2;
    printf("%d\n", *a);
    a = NULL;
}

void test2()
{
    int *a, *b;
    a = (int *)malloc(sizeof(int));
    b = (int *)malloc(sizeof(int));

    if (!(a && b))
    {
        printf("Out of memory\n");
        exit(-1);
    }
    *a = 2;
    *b = 3;
    printf("a: %d, and b: %d\n", *a, *b);

    free(a);
    free(b);
    a = NULL;
    b = NULL;
}

void test3()
{
    int i, *a = (int *)malloc(2000 * sizeof(int));

    if (!a)
    {
        printf("Out of memory\n");
        exit(-1);
    }

    // here the c compiler not use the over-side check, so the code seems right.
    for (i = 0; i < 2000; i++)
    {
        *(i + a) = i;
    }

    free(a);
    a = NULL;
}

void test4()
{
    int **a = (int **)malloc(3 * sizeof(int *));
    for (int i = 0; i < 3; i++)
        a[i] = (int *)malloc(100 * sizeof(int));
    a[1][1] = 5;

    for (int i = 0; i < 3; i++)
        free(a[i]);
    free(a);
    a = NULL;
}

void test5()
{
    int *a = (int *)malloc(sizeof(int));
    printf("Please enter an integer.\n");
    scanf("%d", a);
    if (!*a)
        printf("Value is 0\n");
    else
        printf("%d\n", *a);

    free(a);
    a = NULL;
}

void test6()
{
    struct person
    {
        char gender;
        char name;
        int age;
        float offset;
        double offset_1;
    };

    struct person john;
    john.age = 26;
    john.gender = 'm';
    john.name = 'j';
    john.offset = 234.234;

    printf("%#X\t%#X\t%#X\t%#X\t%#X\t%#X\n", &john, &(john.age), &(john.gender), &(john.name), &(john.offset), &(john.offset_1));
}

int main()
{
    test1();

    test2();

    test3();

    test4();

    test5();

    test6();

    return 0;
}