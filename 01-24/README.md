
**SQL: DB 자격증 취득하는 것도 나쁘지 않음**
* **SQL 구문 학습**
```sql
- 중복 제거
    SELECT DISTINCT Country FROM Customers

- 여러 columns도 가능함
    SELECT DISTINCT City, Country FROM Customers;

- 문자열 선택, 속도 튜닝
    SELECT DISTINCT City, Country 
    FROM Customers
    WHERE City LIKE "%B%" /*WHERE City = "b" */

- 서브 쿼리
    SELECT * 
    FROM (
        SELECT DISTINCT City, Country
        FROM Customers
    ) A
    WHERE A.City LIKE "%B%"

- 별칭 AS는 생략 가능하면 프로그램에 따라서 []또는 ""안에 기술 가능함.

** 집계 빈도 수 상위 5개 조회
    SELECT TOP 5 ContactName, COUNT(*) AS TOTAL
    FROM Suppliers
    GROUP BY ContactName
    ORDER BY COUNT(*) DESC /*TOTAL로 ORDER BY 안됨, TABLE에서 정의한 컬럼이 아니기 때문 */

    SELECT TOP 5 * FROM (
    SELECT ContactName, COUNT(ContactName) AS TOTAL
    FROM Customers
    GROUP BY ContactName
)
    ORDER BY TOTAL DESC /*별칭으로 정의한 변수로 ORDER BY 하기*/
```

#### **SQL 문법의 종류 3가지**
* **데이터 정의 언어 - ( DDL : Data Definition Language )**
* 테이블이나 관계의 구조를 생성하는데 사용하며 CREATE, ALTER, DROP,TRUNCATE 문 등이 있다.

    * **CREATE** - 새로운 데이터베이스 관계 (테이블) View, 인덱스 , 저장 프로시저 만들기.
    * **DROP** - 이미 존재하는 데이터베이스 관계 ( 테이블 ) , 뷰 , 인덱스 , 저장 프로시저를 삭제한다.
    * **ALTER** - 이미 존재하는 데이터베이스 개체에 대한 변경 , RENAME의 역할을 한다.
    * **TRUNCATE** - 관계 ( 테이블 )에서 데이터를 제거한다. ( 한번 삭제시 돌이킬 수 없음.)
* * * 
* **데이터 조작 언어 - ( DML : Data Manipulation Language )** 
    * 테이블에 데이터 검색, 삽입, 수정, 삭제하는 데 사용하며 SELECT, UPDATE, DELETE, INSERT문 등이 있다.

        * **SELECT** - 검색(질의)
        * **INSERT** - 삽입(등록)
        * **UPDATE** - 업데이트(수정)
        * **DELETE** - 삭제

* * *

**데이터 제어 언어 - ( DCL : Data Control Language)**
* 데이터의 사용 권한을 관리하는 데 사용하며 GRANT, REVOKE 문 등이 있다.
 
    * **GRANT** - 특정 데이터베이스 사용자에게 특정 작업에 대한 수행 권한을 부여한다.
    * **REVOKE** - 특정 데이터베이스 사용자에게 특정 작업에 대한 수행 권한을 박탈 or 회수 한다.

    * **데이터베이스 사용자에게 GRANT 및 REVOKE로 설정 할 수 있는 권한**

        * **CONNECT** - 데이터베이스 or 스키마에 연결하는 권한.
        * **SELECT** - 데이터베이스에서 데이터를 검색할 수 있는 권한
        * **INSERT** - 데이터베이스에서 데이터를 등록(삽입)할 수 있는 권한
        * **UPDATE** - 데이터베이스의 데이터를 업데이트 할 수 있는 권한
        * **DELETE** - 데이터베이스의 데이터를 삭제할 수 있는 권한.
        * **USAGE** - 스키마 또는 함수와 같은 데이터베이스 개체를 사용할 수 있는 권한


* 출처: https://zzdd1558.tistory.com/88 [YundleYundle:티스토리]

* **View 테이블**
* 일반적으로 join한 테이블 등록시켜놓음.
    * 또는 만들어놓은 구문의 결과(하위 테이블 대신)
* 일반 user는 테이블 생성 못 하지만 상황에 따라 view는 생성할 수 있음

```sql
# 뷰 테이블 생성

CREATE VIEW view_name AS
SELECT column1, column2, ...
FROM table_name
WHERE condition;
```

### **데이터 무결성**
* 데이터베이스 어트리뷰트는 기본 키(primary key)와 외래 키(foreign key)가 있다. 
* 이때 JOIN이나 SELECT 등의 작업을 할 때, 기본 키는 반드시 중복되면 안 된다. 

**Primary Key(기본키)**
* 후보키 중에서 선택한 주키(Main Key)
* 한 릴레이션에서 특정 튜플을 유일하게 구별할 수 있는 속성
* Null 값을 가질 수 없음
* 기본키로 정의된 속성에는 동일한 값이 중복되어 저장될 수 없음
    * e.g. <학생>릴레이션(테이블)에서 `학번`이나 `주민번호`가 기본키가 될 수 있음,
    * <수강>릴레이션(테이블)에서 `학번`+`과목명`을 조합하여 기본키가 될 수 있음

**Foreign Key(외래키)**
* 관계(Relation)를 맺고 있는 릴레이션 R1, R2에서 릴레이션 R1이 참조하고 있는 릴레이션 R2의 기본키와 같은 R1 릴레이션의 속성

* 외래키로 지정되면 참조 테이블의 기본키에는 없는 값을 입력할 수 없음

    * e.g. <수강>릴레이션(테이블)이 <학생>릴레이션(테이블)을 참조하고 있으므로 <학생>릴레이션의 `학번`은 기본키이고, <수강>릴레이션의 `학번`은 외래키이다
    * <수강>릴레이션의 `학번`에는 <학생>릴레이션의 `학번`에 없는 값을 입력할 수 없다

**`JOIN`과 `WHERE`의 차이점**
<table>
<tr>
<td>일대일(one-to-one)</td>
 <td>일대다(one-to-many)</td>
  <td>다대다(many-to-many)</td>
 </tr> 
  </table>
  
**일대일, 일대다**
* 한 명의 강사는 여러 명의 수강생에게 강의할 수 있음
* 한 명의 수강생은 한 명의 강사의 강의만 들을 수 있음
* 기본키가 중복되는 경우 조회할 수 없음
    * 기본키를 참조하고 있는 값들을 어떻게 한 번에 조회할 것인가에 대한 설계 문제

**다대다**
* 수강생은 여러 강사에게, 강사는 여러 수강생에게

|학습|수강생|강사| 
|:---|:---|:---|
|학습1|수강생1|강사1|
|학습2|수강생2|강사2|
|학습3|수강생3|강사3|
|학습4|수강생4|강사4|
 * 학습: 기본 키

 ```sql
 -- 코드를 입력하세요
# SELECT 
# USER_ID,
# NICKNAME,
# SUM(B.PRICE) AS TOTAL_SALES
# FROM USED_GOODS_BOARD AS B
# JOIN USED_GOODS_USER AS U ON B.WRITER_ID = U.USER_ID
# WHERE B.STATUS = 'DONE'
# GROUP BY U.USER_ID
# HAVING TOTAL_SALES >= 700000
# ORDER BY TOTAL_SALES
 ```
